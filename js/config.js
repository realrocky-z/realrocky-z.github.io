// ============================================
// 个人主页配置文件 —— 只改这个文件即可
// ============================================

var CONFIG = {

    // ---- 基本信息 ----
    name: "Ruihong Zhong",
    bio: "Student at Beijing Normal University, majoring in applied statistics. Interested in data science, deep learning, LLMs and AI. Passionate about learning and sharing knowledge. Hobbies include reading, music, anime, and exploring new technologies. Research Fields: Bayesian Computation, MCMC",

    // ---- 头像（放到 assets/ 文件夹） ----
    avatar: "assets/avatar.png",

    // ---- 经历（按时间倒序） ----
    experience: [
        { year: "Prospective", desc: "Master's student in Statistics, School of Data Science and Engineering, Beijing Normal University"},
        { year: "2023 — Now", desc: "Department of Statistics, Beijing Normal University" },
        { year: "2020 — 2023", desc: "Shenzhen Middle School" }
    ],

    // ---- 社交链接 ----
    links: [
        { type: "github",    href: "https://github.com/realrocky-z",  label: "GitHub" },
        { type: "email",     href: "realrocky0103@outlook.com", label: "Email" },
        { type: "phone",     href: "tel:+86xxxxxxxxxxx",  label: "+86 xxx-xxxx-xxxx" }
    ],

    // ---- Learning 页面：笔记 ----
    notes: [
        {
            title: "Deep Learning Course Notes",
            desc:  "Lecture notes covering neural networks(including DNNs, CNNs, Residual Networks etc.), Optimization techniques in DL, Transformers and generative models(GANs, VAEs, Diffusion Models).",
            tags:  ["Deep Learning", "AI"],
            file:  "files/dl_final.pdf",
            image: "assets/dl.png"
        },
        {
            title: "Research Report on Simplex Method",
            desc:  "Convex Optimization Course Report",
            tags:  ["Convex Optimization", "Computational Mathematics"],
            file:  "files/Research Report on the Simplex Method(20241207 by Rocky).pdf",
            image: "assets/simplex.png"
        },
        {
            title: "Bayesian Statistics Course Notes",
            desc:  "Lecture notes covering Bayesian inference, prior and posterior distributions, conjugate priors, parameter estimation & hypothesis testing, Bayesian model selection and basic Bayesian computation methods.",
            tags:  ["Bayesian Statistics"],
            file:  "files/bayes_stat_course_note_by_rocky.pdf",
            image: "assets/bayesian.png"
        }
    ],

    // ---- Learning 页面：辅导 ----
    tutoring: [
        {
            title: "Statistical Learning — Final Review",
            desc:  "Review slides and practice problems for the Statistical Learning. ",
            tags:  ["2025 Fall"],
            file:  "files/review session1.pdf",
            image: "assets/review session1.png"
        },
        {
            title: "Statistical Computing — Final Review",
            desc:  "Review slides and practice problems for the Statistical Computing.",
            tags:  ["2026 Spring"],
            file:  "files/review session2.pdf",
            image: "assets/review session2.png"
        }
    ],

    // ---- Blog 页面（date 格式 YYYY-MM-DD） ----
    blog: [
        {
            title: "Reading1",
            date:  "2026-10-22",
            desc:  "An essay",
            tags:  ["Essay"],
            file:  "files/quiet-design.pdf",
            image: "assets/writing-1.jpg"
        },
        {
            title: "Reading2",
            date:  "2026-10-23",
            desc:  "Will record something later...",
            tags:  ["Whatever"],
            file:  "files/typography-notes.pdf",
            image: "assets/writing-2.jpg"
        }
    ]

};
