from edtrace import link, text


def article_link(url: str) -> str:
    return link(title="article", url=url)


def post_link(url: str) -> str:
    return link(title="post", url=url)


def video_link(url: str) -> str:
    return link(title="video", url=url)


def bilingual_text(en: str, zh: str, **kwargs):
    """Render English course text followed by its Chinese learning translation.

    渲染英文课程原文，并紧跟用于学习辅助的中文译文。
    """
    return text(f"{en}\n\n{zh}", **kwargs)


def bilingual_note(en: str, zh: str, **kwargs):
    """Render a bilingual explanatory note.

    渲染中英对照的解释性提示。
    """
    return bilingual_text(en, zh, **kwargs)


def bilingual_caption(en: str, zh: str, **kwargs):
    """Render a bilingual figure or table caption.

    渲染中英对照的图片或表格说明。
    """
    return bilingual_text(en, zh, **kwargs)


def bilingual_verbatim(en: str, zh: str, **kwargs):
    """Render an English verbatim block and a Chinese explanation below it.

    英文等宽块保持原样，中文说明单独放在下方，避免破坏 ASCII 对齐。
    """
    text(en, verbatim=True, **kwargs)
    return text(zh)


def get_local_url(path: str) -> str:
    return "https://github.com/stanford-cs336/lectures/blob/main/" + path
