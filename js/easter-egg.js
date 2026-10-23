(function () {
    'use strict';

    // ================================================================
    //  CONFIG — change GOOSE_EMOJI to swap appearance
    //           🪿 goose   🦢 swan   🐔 chicken   🦆 duck   🐤 chick
    // ================================================================
    var GOOSE_EMOJI = '🪿';

    var WALK_SPEED    = 0.6;
    var ALERT_SPEED   = 2.5;
    var FLEE_SPEED    = 4.0;
    var GLOVE_SPEED   = 7;

    var ALERT_DIST       = 230;   // enter alert
    var ALERT_EXIT_DIST  = 360;   // exit alert (hysteresis prevents oscillation)
    var CATCH_DIST       = 50;
    var GLOVE_CATCH_DIST = 50;

    var GLOVE_LIFETIME  = 6000;
    var CAUGHT_DURATION = 2800;
    var FLEE_DURATION   = 2000;

    var IDLE_CHANCE     = 0.08;
    var TARGET_INTERVAL = 3000;
    var SPEECH_COOLDOWN = 4000;

    // ================================================================
    //  SPEECH QUOTES
    // ================================================================
    var quotes = {
        walking: [
            '今天天气真好~',
            '嘎嘎嘎...',
            '好悠闲呀~',
            '这片草地不错...',
            '该往哪走呢?',
            '嗯?有人在看我吗?',
            '啦啦啦~',
            '今天吃啥好呢...',
            '吃饱了散个步~',
            '阳光真暖和呀'
        ],
        alert: [
            '你想干嘛?!',
            '别过来!!',
            '救命啊!!!',
            '我不好吃的!',
            '退!退!退!',
            '啊啊啊别抓我!',
            '妈妈!!!',
            '快跑!!!',
            '离我远点!!',
            '不要过来啊!!'
        ],
        caught: [
            'HONK!!!',
            '放开我!',
            '我错了!',
            '呜呜呜...',
            '手套大哥饶命!',
            '欺负鹅了!!',
            '嘎——!!!',
            '被抓住了!!!'
        ],
        fleeing: [
            '溜了溜了!',
            '跑啊!!!',
            '哈哈抓不到我~',
            '自由了!!',
            '拜拜了您嘞!',
            '好险好险...',
            '下次还敢!',
            '芜湖起飞~'
        ]
    };

    // ================================================================
    //  CREATE DOM
    // ================================================================

    // --- Toggle switch ---
    var toggle = document.createElement('div');
    toggle.className = 'goose-toggle';
    toggle.innerHTML =
        '<span class="goose-toggle-label">召唤大鹅</span>' +
        '<div class="goose-toggle-switch"><div class="goose-toggle-thumb"></div></div>' +
        '<div class="goose-toggle-toast">嘿嘿，并没有</div>';
    document.body.appendChild(toggle);

    var toggleLabel = toggle.querySelector('.goose-toggle-label');
    var toggleToast = toggle.querySelector('.goose-toggle-toast');

    // --- Goose ---
    var goose = document.createElement('div');
    goose.className = 'goose walking';
    goose.innerHTML =
        '<div class="goose-sprite">' + GOOSE_EMOJI + '</div>' +
        '<div class="goose-shadow"></div>' +
        '<div class="goose-speech"></div>';
    document.body.appendChild(goose);

    var sprite   = goose.querySelector('.goose-sprite');
    var speech   = goose.querySelector('.goose-speech');

    // --- Cursor glove ---
    var cursorGlove = document.createElement('div');
    cursorGlove.className = 'cursor-glove';
    cursorGlove.textContent = '🧤';
    document.body.appendChild(cursorGlove);

    // --- Feathers layer ---
    var feathersLayer = document.createElement('div');
    feathersLayer.className = 'feathers-container';
    document.body.appendChild(feathersLayer);

    // ================================================================
    //  STATE
    // ================================================================
    var active   = false;
    var state    = 'walking';
    var pos      = { x: 200, y: 250 };
    var target   = { x: 0, y: 0 };
    var facing   = 1;
    var mousePos = { x: window.innerWidth / 2, y: window.innerHeight / 2 };

    var idleTimer     = null;
    var catchTimer    = null;
    var fleeTimer     = null;
    var intervalId    = null;
    var speechTimer   = null;
    var lastSpeechTime = 0;
    var speechSeq     = 0;

    var gloveEl      = null;
    var glovePos     = { x: 0, y: 0 };
    var gloveActive  = false;
    var gloveExpire  = null;

    var catchCount   = 0;
    var MAX_CATCHES  = 10;

    // ================================================================
    //  TOGGLE LOGIC
    // ================================================================
    var toggleClicks  = 0;
    var toggleTimer   = null;

    toggle.addEventListener('click', function (e) {
        e.stopPropagation();
        e.preventDefault();

        toggleClicks++;
        clearTimeout(toggleTimer);

        if (toggleClicks === 1) {
            toggleLabel.classList.add('stage-1');

            toggleTimer = setTimeout(function () {
                // Too slow — show joke toast and reset
                if (toggleClicks < 3 && !active) {
                    showToast();
                    resetToggleLabel();
                }
                toggleClicks = 0;
            }, 600);
        } else if (toggleClicks === 2) {
            toggleLabel.classList.remove('stage-1');
            toggleLabel.classList.add('stage-2');

            toggleTimer = setTimeout(function () {
                if (toggleClicks < 3 && !active) {
                    showToast();
                    resetToggleLabel();
                }
                toggleClicks = 0;
            }, 600);
        } else if (toggleClicks >= 3) {
            clearTimeout(toggleTimer);
            toggleClicks = 0;
            if (!active) {
                activateGoose();
            }
        }
    });

    function resetToggleLabel() {
        toggleLabel.classList.remove('stage-1', 'stage-2');
    }

    function showToast() {
        toggleToast.classList.add('show');
        setTimeout(function () {
            toggleToast.classList.remove('show');
        }, 2000);
    }

    function activateGoose() {
        active = true;
        catchCount = 0;
        toggle.classList.add('active');
        toggleLabel.textContent = '已召唤';
        resetToggleLabel();

        goose.classList.add('active');

        // Position goose near bottom-right of toggle
        pos.x = window.innerWidth - 150;
        pos.y = 80;
        goose.style.left = pos.x + 'px';
        goose.style.top = pos.y + 'px';

        pickTarget();
        startLoop();

        // Restart target-pick interval (cleared by resetAll)
        if (intervalId) clearInterval(intervalId);
        intervalId = setInterval(function () {
            if (state === 'walking') pickTarget();
        }, TARGET_INTERVAL);
    }

    // ================================================================
    //  HELPERS
    // ================================================================
    function setState(s) {
        state = s;
        goose.className = 'goose active ' + s;
    }

    function pickTarget() {
        var m = 100;
        target.x = m + Math.random() * (window.innerWidth - m * 2);
        target.y = m + Math.random() * (window.innerHeight - m * 2);
    }

    function maybeIdle() {
        if (state === 'walking' && Math.random() < IDLE_CHANCE) {
            setState('idle');
            clearTimeout(idleTimer);
            idleTimer = setTimeout(function () {
                if (state === 'idle') {
                    setState('walking');
                    pickTarget();
                }
            }, 1200 + Math.random() * 2000);
        }
    }

    function setAlertTarget() {
        var angle = Math.atan2(pos.y - mousePos.y, pos.x - mousePos.x);
        target.x = pos.x + Math.cos(angle) * 400;
        target.y = pos.y + Math.sin(angle) * 400;
        target.x = Math.max(60, Math.min(window.innerWidth - 60, target.x));
        target.y = Math.max(60, Math.min(window.innerHeight - 60, target.y));
    }

    function dist(a, b) {
        var dx = a.x - b.x;
        var dy = a.y - b.y;
        return Math.sqrt(dx * dx + dy * dy);
    }

    // ================================================================
    //  SPEECH
    // ================================================================
    function showSpeech(category, force) {
        var now = Date.now();
        if (!force && now - lastSpeechTime < SPEECH_COOLDOWN) return;
        lastSpeechTime = now;

        var list = quotes[category];
        if (!list || !list.length) return;
        var text = list[Math.floor(Math.random() * list.length)];

        speech.classList.remove('show');
        speechSeq++;
        var seq = speechSeq;

        setTimeout(function () {
            if (seq !== speechSeq) return;
            speech.textContent = text;
            speech.classList.add('show');
            clearTimeout(speechTimer);
            speechTimer = setTimeout(function () {
                speech.classList.remove('show');
            }, 3000);
        }, 80);
    }

    // ================================================================
    //  FEATHERS
    // ================================================================
    function spawnFeathers(n) {
        var i, f, angle, d;
        for (i = 0; i < n; i++) {
            f = document.createElement('div');
            f.className = 'feather';
            f.style.left = pos.x + 'px';
            f.style.top = pos.y + 'px';
            angle = (i / n) * Math.PI * 2 + (Math.random() - 0.5) * 0.8;
            d = 30 + Math.random() * 55;
            f.style.setProperty('--dx', Math.cos(angle) * d + 'px');
            f.style.setProperty('--dy', Math.sin(angle) * d + 'px');
            f.style.setProperty('--rot', (Math.random() * 720 - 360) + 'deg');
            feathersLayer.appendChild(f);
            setTimeout(function (el) { return function () { el.remove(); }; }(f), 650);
        }
    }

    // ================================================================
    //  SUMMONED GLOVE (double-click)
    // ================================================================
    function cleanupGlove() {
        gloveActive = false;
        if (gloveEl) { gloveEl.remove(); gloveEl = null; }
        clearTimeout(gloveExpire);
        gloveExpire = null;
    }

    function summonGlove(cx, cy) {
        if (gloveActive || state === 'caught') return;
        cleanupGlove();

        gloveActive = true;
        glovePos.x = cx;
        glovePos.y = cy;

        gloveEl = document.createElement('div');
        gloveEl.className = 'glove';
        gloveEl.textContent = '🧤';
        gloveEl.style.left = cx + 'px';
        gloveEl.style.top = cy + 'px';
        document.body.appendChild(gloveEl);

        gloveExpire = setTimeout(function () {
            if (gloveActive && state !== 'caught') {
                if (gloveEl) gloveEl.classList.add('fading');
                setTimeout(cleanupGlove, 400);
            }
        }, GLOVE_LIFETIME);
    }

    // ================================================================
    //  CATCH & ESCAPE
    // ================================================================
    function catchGoose() {
        if (state === 'caught') return;

        catchCount++;
        if (catchCount >= MAX_CATCHES) {
            finalCatch();
            return;
        }

        setState('caught');
        gloveActive = false;
        clearTimeout(gloveExpire);
        gloveExpire = null;

        if (gloveEl) {
            gloveEl.classList.add('grabbing');
            gloveEl.style.left = (pos.x - 4) + 'px';
            gloveEl.style.top = (pos.y - 24) + 'px';
        }

        showSpeech('caught', true);
        spawnFeathers(12);

        clearTimeout(catchTimer);
        catchTimer = setTimeout(escapeGoose, CAUGHT_DURATION);
    }

    function finalCatch() {
        setState('caught');
        gloveActive = false;
        clearTimeout(gloveExpire);
        gloveExpire = null;

        if (gloveEl) {
            gloveEl.classList.add('grabbing');
            gloveEl.style.left = (pos.x - 4) + 'px';
            gloveEl.style.top = (pos.y - 24) + 'px';
        }

        // Dramatic feather burst
        spawnFeathers(24);

        // Initial "HONK!" then switch to ending text
        showSpeech('caught', true);
        setTimeout(function () {
            speech.classList.remove('show');
            speechSeq++;
            var seq = speechSeq;
            setTimeout(function () {
                if (seq !== speechSeq) return;
                var endings = ['鹅已经被抓完了...', '鹅累了，休息一下吧......'];
                speech.textContent = endings[Math.floor(Math.random() * endings.length)];
                speech.classList.add('show');
            }, 150);
        }, 1200);

        // Fade out goose
        setTimeout(function () {
            goose.classList.add('fading-out');
        }, 3000);

        // Full reset after fade completes
        setTimeout(function () {
            resetAll();
        }, 5000);
    }

    function resetAll() {
        if (loopId) {
            cancelAnimationFrame(loopId);
            loopId = null;
        }

        // Remove all state classes, revert to hidden walking
        goose.classList.remove('active', 'fading-out', 'walking', 'alert', 'caught', 'fleeing', 'idle');
        goose.className = 'goose walking';
        speech.classList.remove('show');
        speech.textContent = '';

        // Reset state
        state = 'walking';
        active = false;
        catchCount = 0;
        toggle.classList.remove('active');
        toggleLabel.textContent = '召唤大鹅';
        resetToggleLabel();

        // Cleanup glove & cursor
        cleanupGlove();
        document.body.style.cursor = '';
        cursorGlove.style.opacity = '0';
        cursorGlove.classList.remove('visible');

        // Clear all timers
        clearTimeout(catchTimer);
        catchTimer = null;
        clearTimeout(fleeTimer);
        fleeTimer = null;
        clearTimeout(idleTimer);
        idleTimer = null;
        clearTimeout(speechTimer);
        speechTimer = null;
        clearInterval(intervalId);
        intervalId = null;
    }

    function escapeGoose() {
        setState('fleeing');
        if (gloveEl) {
            gloveEl.classList.add('releasing');
            setTimeout(function () {
                if (gloveEl && !gloveActive) { gloveEl.remove(); gloveEl = null; }
            }, 350);
        }

        showSpeech('fleeing', true);
        pickTarget();

        clearTimeout(fleeTimer);
        fleeTimer = setTimeout(function () {
            if (state === 'fleeing') {
                var d = dist(pos, mousePos);
                setState(d < ALERT_DIST ? 'alert' : 'walking');
                if (d >= ALERT_DIST) pickTarget();
            }
        }, FLEE_DURATION);
    }

    // ================================================================
    //  MAIN LOOP
    // ================================================================
    var loopId = null;

    function loop() {
        var distToMouse = dist(pos, mousePos);

        var hovered = null;
        var isInteractive = false;
        if (distToMouse < ALERT_DIST) {
            hovered = document.elementFromPoint(mousePos.x, mousePos.y);
            isInteractive = !!(hovered && hovered.closest('button, a, input, .modal-overlay'));
        }

        // State transitions from mouse proximity
        if (state !== 'caught' && state !== 'fleeing') {
            if (distToMouse < CATCH_DIST && !isInteractive) {
                catchGoose();
            } else if (distToMouse < ALERT_DIST && !isInteractive) {
                if (state !== 'alert') {
                    setState('alert');
                    showSpeech('alert', true);
                    setAlertTarget();
                }
                // Target is only set once on entering alert;
                // goose commits to that direction until reaching it.
            } else if (state === 'alert' && distToMouse > ALERT_EXIT_DIST) {
                setState('walking');
                pickTarget();
            }
        }

        // Cursor glove
        var showGlove = distToMouse < ALERT_DIST && state !== 'caught' && !isInteractive;
        cursorGlove.style.opacity = showGlove ? '1' : '0';
        if (showGlove) {
            cursorGlove.classList.add('visible');
        } else {
            cursorGlove.classList.remove('visible');
        }

        // Cursor style
        if (distToMouse < CATCH_DIST && state !== 'caught' && state !== 'fleeing' && !isInteractive) {
            document.body.style.cursor = 'grabbing';
        } else if (distToMouse < ALERT_DIST && state !== 'caught' && !isInteractive) {
            document.body.style.cursor = 'grab';
        } else {
            document.body.style.cursor = '';
        }

        // Move goose
        if (state === 'walking' || state === 'alert' || state === 'fleeing') {
            var speed;
            if (state === 'fleeing') speed = FLEE_SPEED;
            else if (state === 'alert') speed = ALERT_SPEED;
            else speed = WALK_SPEED;

            var dx = target.x - pos.x;
            var dy = target.y - pos.y;
            var d = Math.sqrt(dx * dx + dy * dy);

            if (d < 6) {
                if (state === 'walking') {
                    pickTarget();
                    maybeIdle();
                } else if (state === 'alert') {
                    // Reached flee target — reassess
                    if (dist(pos, mousePos) < ALERT_EXIT_DIST) {
                        setAlertTarget();
                    } else {
                        setState('walking');
                        pickTarget();
                    }
                }
            } else {
                pos.x += (dx / d) * speed;
                pos.y += (dy / d) * speed;
                if (Math.abs(dx) > 1.5) {
                    facing = dx > 0 ? 1 : -1;
                }
            }
        }

        goose.style.left = pos.x + 'px';
        goose.style.top = pos.y + 'px';
        goose.style.setProperty('--facing', facing);
        speech.style.setProperty('--fix', facing);

        // Move summoned glove
        if (gloveActive && gloveEl && state !== 'caught') {
            var gd = dist(glovePos, pos);
            if (gd < GLOVE_CATCH_DIST) {
                catchGoose();
            } else {
                var gdx = pos.x - glovePos.x;
                var gdy = pos.y - glovePos.y;
                glovePos.x += (gdx / gd) * GLOVE_SPEED;
                glovePos.y += (gdy / gd) * GLOVE_SPEED;
                gloveEl.style.left = glovePos.x + 'px';
                gloveEl.style.top = glovePos.y + 'px';
            }
        }

        loopId = requestAnimationFrame(loop);
    }

    function startLoop() {
        if (loopId) return;
        loopId = requestAnimationFrame(loop);
    }

    // ================================================================
    //  EVENTS
    // ================================================================
    document.addEventListener('mousemove', function (e) {
        mousePos.x = e.clientX;
        mousePos.y = e.clientY;
        cursorGlove.style.left = (e.clientX - 15) + 'px';
        cursorGlove.style.top  = (e.clientY - 15) + 'px';
    });

    document.addEventListener('dblclick', function (e) {
        if (!active) return;
        if (e.target.closest('button, a, input, .modal-overlay, .goose-toggle')) return;
        e.preventDefault();
        summonGlove(e.clientX, e.clientY);
    });

    // Mobile: triple-tap
    var tapCount = 0;
    var tapReset = null;
    document.addEventListener('click', function (e) {
        if (!active) return;
        if (e.target.closest('button, a, input, .modal-overlay, .goose-toggle')) { tapCount = 0; return; }
        tapCount++;
        if (tapCount === 1) {
            tapReset = setTimeout(function () { tapCount = 0; }, 500);
        } else if (tapCount === 3) {
            clearTimeout(tapReset);
            tapCount = 0;
            summonGlove(e.clientX, e.clientY);
        }
    });

    window.addEventListener('resize', function () {
        pos.x = Math.min(pos.x, window.innerWidth - 30);
        pos.y = Math.min(pos.y, window.innerHeight - 30);
    });

    // ================================================================
    //  TOGGLE VISIBILITY — only show on Home page
    // ================================================================
    toggle.style.display = 'none';

    function updateToggleVisibility() {
        var homePage = document.getElementById('page-home');
        toggle.style.display = (homePage && homePage.classList.contains('active')) ? '' : 'none';
    }

    updateToggleVisibility();

    // Hook into existing nav tab clicks (delegated from main.js pattern)
    document.querySelectorAll('.nav-tab').forEach(function (tab) {
        tab.addEventListener('click', function () {
            setTimeout(updateToggleVisibility, 60);
        });
    });

    // ================================================================
    //  PERIODIC WALKING SPEECH (only when active)
    // ================================================================
    setInterval(function () {
        if (active && state === 'walking' && Math.random() < 0.35) {
            showSpeech('walking');
        }
    }, 10000);

})();
