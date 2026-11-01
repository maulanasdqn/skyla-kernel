use crate::socket::Link;
use crate::state::Chat;
use leptos::prelude::*;
use skyla_types::{ClientMessage, ModelEntry};

fn label(m: &ModelEntry) -> String {
    let status = if m.supported {
        String::new()
    } else {
        format!(" · {} not supported", m.arch)
    };
    format!("{} ({:.1} GB){status}", m.name, m.size_gb)
}

#[component]
pub fn ModelPicker() -> impl IntoView {
    let chat = expect_context::<Chat>();
    let link = expect_context::<Link>();
    let current = move || {
        chat.info
            .with(|i| i.as_ref().map(|i| i.model_id.clone()).unwrap_or_default())
    };
    let locked = move || chat.busy.get() || chat.loading.with(Option::is_some) || !chat.connected.get();

    view! {
        <select
            class="picker"
            disabled=locked
            prop:value=current
            on:change=move |ev| {
                let id = event_target_value(&ev);
                if id != current() {
                    link.send(ClientMessage::Load { id });
                }
            }
        >
            <For
                each=move || chat.models.get()
                key=|m| m.id.clone()
                children=move |m| {
                    let id = m.id.clone();
                    view! {
                        <option value=id.clone() disabled=!m.supported selected=move || current() == id>
                            {label(&m)}
                        </option>
                    }
                }
            />
        </select>
    }
}
