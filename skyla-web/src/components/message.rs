use crate::state::{Chat, Turn};
use leptos::prelude::*;
use skyla_types::Stats;

fn describe(s: &Stats) -> String {
    format!(
        "{} tokens · {:.2} tok/s · prompt {} tokens in {:.0} ms · context {}/{}",
        s.tokens, s.tokens_per_second, s.prompt_tokens, s.prefill_ms, s.position, s.context
    )
}

#[component]
pub fn MessageView(turn: Turn) -> impl IntoView {
    if turn.user {
        return view! {
            <div class="turn user">
                <div class="bubble">{move || turn.text.get()}</div>
            </div>
        }
        .into_any();
    }

    let chat = expect_context::<Chat>();
    let has_thinking = move || turn.thinking.with(|t| !t.trim().is_empty());
    let answering = move || turn.text.with(|t| !t.is_empty());
    let finished = move || turn.stats.with(Option::is_some) || turn.error.with(Option::is_some);
    let label = move || match !answering() && !finished() && chat.busy.get() {
        true => "Thinking…".to_string(),
        false => format!(
            "Thought for {} words",
            turn.thinking.with(|t| t.split_whitespace().count())
        ),
    };

    view! {
        <div class="turn assistant">
            <Show when=has_thinking>
                <details class="thinking" open=true>
                    <summary>{label}</summary>
                    <div class="thought">{move || turn.thinking.get()}</div>
                </details>
            </Show>
            <Show when=move || !has_thinking() && !answering() && !finished()>
                <div class="typing">
                    <span></span>
                    <span></span>
                    <span></span>
                </div>
            </Show>
            <div class="answer">{move || turn.text.get()}</div>
            {move || turn.error.get().map(|e| view! { <div class="error">{e}</div> })}
            {move || turn.stats.get().map(|s| view! { <div class="stats">{describe(&s)}</div> })}
        </div>
    }
    .into_any()
}
