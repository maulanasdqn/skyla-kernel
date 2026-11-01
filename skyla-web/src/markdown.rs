use pulldown_cmark::{html, CowStr, Event, Options, Parser, Tag};

pub fn render(text: &str) -> String {
    let options = Options::ENABLE_TABLES | Options::ENABLE_STRIKETHROUGH | Options::ENABLE_TASKLISTS;
    let events = Parser::new_ext(text, options).map(|e| match e {
        Event::Html(raw) | Event::InlineHtml(raw) => Event::Text(raw),
        Event::Start(Tag::Link {
            link_type,
            dest_url,
            title,
            id,
        }) if !safe(&dest_url) => Event::Start(Tag::Link {
            link_type,
            dest_url: CowStr::Borrowed("#"),
            title,
            id,
        }),
        Event::Start(Tag::Image {
            link_type,
            dest_url,
            title,
            id,
        }) if !safe(&dest_url) => Event::Start(Tag::Image {
            link_type,
            dest_url: CowStr::Borrowed("#"),
            title,
            id,
        }),
        other => other,
    });
    let mut out = String::with_capacity(text.len() * 3 / 2);
    html::push_html(&mut out, events);
    out
}

fn safe(url: &str) -> bool {
    let lower = url.trim_start().to_ascii_lowercase();
    !lower.contains(':') || ["http:", "https:", "mailto:"].iter().any(|s| lower.starts_with(s))
}
