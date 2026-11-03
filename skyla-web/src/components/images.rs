use crate::images::Studio;
use leptos::prelude::*;
use skyla_types::ImageStatus;

#[component]
pub fn ImagesView() -> impl IntoView {
    let studio = expect_context::<Studio>();
    studio.refresh();
    let ready = move || !studio.model.with(String::is_empty) && !studio.prompt.with(|p| p.trim().is_empty());

    view! {
        <div class="images">
            <div class="image-models">
                <For
                    each=move || studio.models.get()
                    key=|m| m.id.clone()
                    children=move |m| {
                        let id = m.id.clone();
                        let selected = {
                            let id = id.clone();
                            move || studio.model.with(|s| *s == id)
                        };
                        let detail = match (m.cloud, m.xl) {
                            (true, _) => "☁️ cloud · prompt is sent to fal.ai",
                            (false, true) => "1024 × 1024 · on this Mac",
                            (false, false) => "512 × 512 · fast · on this Mac",
                        };
                        view! {
                            <button
                                class="model-card"
                                class:selected=selected
                                disabled=move || studio.busy()
                                on:click=move |_| studio.model.set(id.clone())
                            >
                                <strong>{m.name.clone()}</strong>
                                <span>{detail}</span>
                            </button>
                        }
                    }
                />
            </div>
            <Show when=move || studio.models.with(Vec::is_empty)>
                <p class="detail">"No image models found. Put Core ML Stable Diffusion folders in ~/sd-models."</p>
            </Show>
            <textarea
                class="prompt"
                rows="3"
                placeholder="Describe the image… (English works best)"
                prop:value=move || studio.prompt.get()
                on:input=move |ev| studio.prompt.set(event_target_value(&ev))
            ></textarea>
            <details class="advanced">
                <summary>"Advanced"</summary>
                <label>"Negative prompt"
                    <textarea
                        rows="2"
                        prop:value=move || studio.negative.get()
                        on:input=move |ev| studio.negative.set(event_target_value(&ev))
                    ></textarea>
                </label>
                <label>{move || format!("Steps {}", studio.steps.get())}
                    <input type="range" min="8" max="40" step="1"
                        prop:value=move || studio.steps.get().to_string()
                        on:input=move |ev| studio.steps.set(event_target_value(&ev).parse().unwrap_or(20)) />
                </label>
                <label>{move || format!("Guidance {:.1}", studio.guidance.get())}
                    <input type="range" min="2" max="12" step="0.5"
                        prop:value=move || studio.guidance.get().to_string()
                        on:input=move |ev| studio.guidance.set(event_target_value(&ev).parse().unwrap_or(7.0)) />
                </label>
            </details>
            <Progress/>
            <Show when=move || !studio.busy()>
                <button class="generate" disabled=move || !ready() on:click=move |_| studio.generate()>
                    "✨ Generate"
                </button>
            </Show>
            {move || studio.problem.get().map(|p| view! { <p class="error">{p}</p> })}
            <Latest/>
            <Gallery/>
        </div>
    }
}

#[component]
fn Progress() -> impl IntoView {
    let studio = expect_context::<Studio>();
    let state = move || match studio.status.get() {
        ImageStatus::Starting => Some(("Starting…".to_string(), None)),
        ImageStatus::Loading => Some(("Loading the model… the first run takes longer".to_string(), None)),
        ImageStatus::Remote => Some(("Generating on fal.ai…".to_string(), None)),
        ImageStatus::Step { step, total } => Some((
            format!("Step {step} of {total}"),
            Some(step as f64 / total.max(1) as f64),
        )),
        _ => None,
    };
    move || {
        state().map(|(label, value)| {
            view! {
                <div class="progress">
                    <div class="progress-row">
                        <span>{label}</span>
                        <button class="ghost" on:click=move |_| studio.cancel()>"Cancel"</button>
                    </div>
                    <div class="bar">
                        <div class="fill" class:indeterminate=value.is_none()
                            style:width=format!("{}%", value.unwrap_or(1.0) * 100.0)></div>
                    </div>
                </div>
            }
        })
    }
}

#[component]
fn Latest() -> impl IntoView {
    let studio = expect_context::<Studio>();
    move || match studio.status.get() {
        ImageStatus::Done {
            image,
            seconds,
            sampling_seconds,
        } => Some(view! {
            <figure class="latest">
                <a href=image.url.clone() target="_blank"><img src=image.url.clone() /></a>
                <figcaption>
                    {format!("{:.1} s total · {:.1} s sampling", seconds, sampling_seconds)}
                    " · "
                    <a href=image.url.clone() download=image.name.clone()>"Download"</a>
                </figcaption>
            </figure>
        }),
        _ => None,
    }
}

#[component]
fn Gallery() -> impl IntoView {
    let studio = expect_context::<Studio>();
    view! {
        <Show when=move || !studio.gallery.with(Vec::is_empty)>
            <h3>"Gallery"</h3>
            <div class="gallery">
                <For
                    each=move || studio.gallery.get()
                    key=|g| g.name.clone()
                    children=move |g| view! {
                        <a href=g.url.clone() target="_blank"><img src=g.url.clone() loading="lazy" /></a>
                    }
                />
            </div>
        </Show>
    }
}
