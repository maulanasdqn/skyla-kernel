use crate::state::Chat;
use leptos::prelude::*;

#[component]
pub fn SettingsPanel() -> impl IntoView {
    let settings = expect_context::<Chat>().settings;

    view! {
        <section class="settings">
            <label>
                <span>"Temperature " {move || format!("{:.2}", settings.with(|s| s.temperature))}</span>
                <input
                    type="range"
                    min="0"
                    max="1.5"
                    step="0.05"
                    prop:value=move || settings.with(|s| s.temperature.to_string())
                    on:input=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                            settings.update(|s| s.temperature = v);
                        }
                    }
                />
            </label>
            <label>
                <span>"Top-p " {move || format!("{:.2}", settings.with(|s| s.top_p))}</span>
                <input
                    type="range"
                    min="0.1"
                    max="1"
                    step="0.05"
                    prop:value=move || settings.with(|s| s.top_p.to_string())
                    on:input=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<f32>() {
                            settings.update(|s| s.top_p = v);
                        }
                    }
                />
            </label>
            <label>
                <span>"Max tokens per reply"</span>
                <input
                    type="number"
                    min="64"
                    max="16384"
                    step="64"
                    prop:value=move || settings.with(|s| s.max_tokens.to_string())
                    on:change=move |ev| {
                        if let Ok(v) = event_target_value(&ev).parse::<usize>() {
                            settings.update(|s| s.max_tokens = v.clamp(64, 16384));
                        }
                    }
                />
            </label>
            <label class="wide">
                <span>"System prompt (applies to a new chat)"</span>
                <textarea
                    rows="2"
                    prop:value=move || settings.with(|s| s.system.clone().unwrap_or_default())
                    on:input=move |ev| {
                        let v = event_target_value(&ev);
                        settings.update(|s| s.system = (!v.trim().is_empty()).then_some(v));
                    }
                ></textarea>
            </label>
        </section>
    }
}
