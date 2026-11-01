use super::picker::ModelPicker;
use super::settings::SettingsPanel;
use crate::socket::Link;
use crate::state::Chat;
use leptos::prelude::*;
use skyla_types::ClientMessage;

#[component]
pub fn Header() -> impl IntoView {
    let chat = expect_context::<Chat>();
    let link = expect_context::<Link>();
    let open = RwSignal::new(false);

    let detail = move || {
        if let Some(id) = chat.loading.get() {
            return format!("loading {id} …");
        }
        chat.info.with(|i| match i {
            Some(i) => format!(
                "{} · {} / {} tokens · KV {}",
                i.device, i.position, i.context, i.kv_cache
            ),
            None if chat.connected.get() => "connecting…".into(),
            None => "offline: start `skyla serve <model>`".into(),
        })
    };
    let status = move || {
        if chat.connected.get() {
            "dot online"
        } else {
            "dot offline"
        }
    };

    view! {
        <header class="header">
            <div class="brand">
                <span class=status></span>
                <div>
                    <ModelPicker/>
                    <div class="detail">{detail}</div>
                </div>
            </div>
            <div class="actions">
                <button class="ghost" on:click=move |_| open.update(|o| *o = !*o)>
                    "Settings"
                </button>
                <button
                    class="ghost"
                    disabled=move || !chat.connected.get()
                    on:click=move |_| link.send(ClientMessage::Reset)
                >
                    "New chat"
                </button>
            </div>
        </header>
        <Show when=move || open.get()>
            <SettingsPanel/>
        </Show>
    }
}
