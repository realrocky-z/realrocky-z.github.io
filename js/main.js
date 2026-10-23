// ===== Page switching =====
(function () {
    var tabs = document.querySelectorAll('.nav-tab');
    var pages = document.querySelectorAll('.page');

    tabs.forEach(function (tab) {
        tab.addEventListener('click', function () {
            var target = this.getAttribute('data-page');
            tabs.forEach(function (t) { t.classList.remove('active'); });
            this.classList.add('active');
            pages.forEach(function (p) {
                p.classList.remove('active');
                if (p.id === 'page-' + target) {
                    p.classList.add('active');
                    p.style.animation = 'none';
                    p.offsetHeight;
                    p.style.animation = '';
                }
            });
        });
    });
})();

// ===== Render home page from CONFIG =====
(function () {
    var C = CONFIG;

    // Document title & brand
    document.title = C.name + ' — Personal Site';
    var brand = document.querySelector('.nav-brand');
    if (brand) brand.textContent = C.name;

    // Avatar with fallback
    var avatar = document.getElementById('avatar-img');
    var fallback = document.querySelector('.avatar-fallback');
    if (fallback) {
        fallback.textContent = C.name.split(' ').map(function (w) { return w[0]; }).join('').toUpperCase().slice(0, 2);
    }
    if (avatar && C.avatar) {
        function showImage() {
            avatar.style.display = '';
            if (fallback) fallback.style.display = 'none';
        }
        function showFallback() {
            avatar.style.display = 'none';
            if (fallback) fallback.style.display = 'flex';
        }
        avatar.onerror = showFallback;
        avatar.onload = showImage;
        avatar.src = C.avatar;
        // Handle cached image (already loaded before onload set)
        if (avatar.complete && avatar.naturalWidth > 0) {
            showImage();
        } else if (avatar.complete && avatar.naturalWidth === 0) {
            showFallback();
        }
    } else if (fallback) {
        fallback.style.display = 'flex';
    }

    // Name & bio
    var nameEl = document.getElementById('home-name');
    if (nameEl) nameEl.textContent = C.name;
    var bioEl = document.getElementById('home-bio');
    if (bioEl) bioEl.textContent = C.bio;

    // Experience
    var expList = document.getElementById('home-experience');
    if (expList && C.experience) {
        var html = '';
        C.experience.forEach(function (item) {
            html += '<div class="exp-item">'
                  + '<span class="exp-year">' + escapeHTML(item.year) + '</span>'
                  + '<span class="exp-desc">' + escapeHTML(item.desc) + '</span>'
                  + '</div>';
        });
        expList.innerHTML = html;
    }

    // Social links
    var linksEl = document.getElementById('home-links');
    if (linksEl && C.links) {
        var html = '';
        C.links.forEach(function (link) {
            var svg = getIcon(link.type);
            var cls = link.type === 'phone' ? 'link-phone' : '';
            var tooltip = link.type === 'phone' ? ' data-tooltip="' + escapeAttr(link.label) + '"' : '';
            var title = link.type !== 'phone' ? ' title="' + escapeAttr(link.label) + '"' : '';
            html += '<a href="' + escapeAttr(link.href) + '" class="' + cls + '"' + tooltip + title + ' target="_blank" rel="noopener">' + svg + '</a>';
        });
        linksEl.innerHTML = html;
    }

    // Footer
    var footer = document.querySelector('.footer p');
    if (footer) {
        footer.innerHTML = '&copy; ' + new Date().getFullYear() + ' ' + escapeHTML(C.name);
    }
})();

// ===== Render Shelf (Learning) page from CONFIG =====
(function () {
    var notesGrid = document.getElementById('notes-grid');
    if (notesGrid && CONFIG.notes) {
        notesGrid.innerHTML = renderCards(CONFIG.notes);
    }
    var tutoringGrid = document.getElementById('tutoring-grid');
    if (tutoringGrid && CONFIG.tutoring) {
        tutoringGrid.innerHTML = renderCards(CONFIG.tutoring);
    }
})();

// ===== Render Blog page from CONFIG =====
(function () {
    var list = document.getElementById('blog-list');
    var select = document.getElementById('blog-sort');
    if (!list || !CONFIG.blog) return;

    function render(order) {
        var items = CONFIG.blog.slice().sort(function (a, b) {
            var da = a.date || '';
            var db = b.date || '';
            if (da === db) return 0;
            var cmp = da < db ? -1 : 1;
            return order === 'asc' ? cmp : -cmp;
        });

        var html = '';
        items.forEach(function (item) {
            var tags = '';
            item.tags.forEach(function (t) { tags += '<span>' + escapeHTML(t) + '</span>'; });
            html += '<article class="blog-row">'
                  + '<time class="blog-date" datetime="' + escapeAttr(item.date || '') + '">' + escapeHTML(item.date || '—') + '</time>'
                  + '<div class="blog-body">'
                  + '<h3>' + escapeHTML(item.title) + '</h3>'
                  + '<p>' + escapeHTML(item.desc) + '</p>'
                  + '<div class="card-tags">' + tags + '</div>'
                  + '</div>'
                  + '<button class="btn-download" data-file="' + escapeAttr(item.file) + '" data-label="' + escapeAttr(item.title) + '">Download</button>'
                  + '</article>';
        });
        list.innerHTML = html;
    }

    render(select ? select.value : 'desc');
    if (select) {
        select.addEventListener('change', function () { render(this.value); });
    }
})();

// ===== Downloads (PDFs are password-protected at file level) =====
(function () {
    document.addEventListener('click', function (e) {
        var btn = e.target.closest('.btn-download');
        if (!btn) return;
        e.preventDefault();
        var a = document.createElement('a');
        a.href = btn.getAttribute('data-file');
        a.download = '';
        a.style.display = 'none';
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
    });
})();

// ===== Helpers =====
function renderCards(items) {
    var html = '';
    items.forEach(function (item) {
        var tags = '';
        item.tags.forEach(function (t) { tags += '<span>' + escapeHTML(t) + '</span>'; });
        html += '<article class="card">'
              + '<div class="card-preview"><img src="' + escapeAttr(item.image) + '" alt="' + escapeAttr(item.title) + '" onerror="this.style.display=\'none\'"></div>'
              + '<div class="card-body">'
              + '<h3>' + escapeHTML(item.title) + '</h3>'
              + '<p>' + escapeHTML(item.desc) + '</p>'
              + '<div class="card-tags">' + tags + '</div>'
              + '<button class="btn-download" data-file="' + escapeAttr(item.file) + '" data-label="' + escapeAttr(item.title) + '">Download file</button>'
              + '</div></article>';
    });
    return html;
}

function escapeHTML(str) {
    return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

function escapeAttr(str) {
    return String(str).replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// SVG icons
function getIcon(type) {
    switch (type) {
        case 'github':
            return '<svg viewBox="0 0 24 24" width="20" height="20" fill="currentColor"><path d="M12 0C5.37 0 0 5.37 0 12c0 5.31 3.435 9.795 8.205 11.385.6.105.825-.255.825-.57 0-.285-.015-1.23-.015-2.235-3.015.555-3.795-.735-4.035-1.41-.135-.345-.72-1.41-1.23-1.695-.42-.225-1.02-.78-.015-.795.945-.015 1.62.87 1.845 1.23 1.08 1.815 2.805 1.305 3.495.99.105-.78.42-1.305.765-1.605-2.67-.3-5.46-1.335-5.46-5.925 0-1.305.465-2.385 1.23-3.225-.12-.3-.54-1.53.12-3.18 0 0 1.005-.315 3.3 1.23.96-.27 1.98-.405 3-.405s2.04.135 3 .405c2.295-1.56 3.3-1.23 3.3-1.23.66 1.65.24 2.88.12 3.18.765.84 1.23 1.905 1.23 3.225 0 4.605-2.805 5.625-5.475 5.925.435.375.81 1.095.81 2.22 0 1.605-.015 2.895-.015 3.3 0 .315.225.69.825.57A12.02 12.02 0 0 0 24 12c0-6.63-5.37-12-12-12z"/></svg>';
        case 'email':
            return '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="4" width="20" height="16" rx="2"/><path d="M22 4L12 13 2 4"/></svg>';
        case 'phone':
            return '<svg viewBox="0 0 24 24" width="20" height="20" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.127.96.361 1.903.7 2.81a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.907.339 1.85.573 2.81.7A2 2 0 0 1 22 16.92z"/></svg>';
        default:
            return '';
    }
}
