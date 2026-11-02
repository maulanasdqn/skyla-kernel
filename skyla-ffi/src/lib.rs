use skyla_chat::{seed_from_clock, Piece, Sampler, Session};
use skyla_engine::KvPrecision;
use skyla_gguf::Gguf;
use skyla_tok::Tokenizer;
use skyla_types::Settings;
use std::ffi::{c_char, c_void, CStr, CString};
use std::mem::ManuallyDrop;
use std::ptr;
use std::sync::atomic::{AtomicBool, Ordering};

pub type TokenCallback = extern "C" fn(user: *mut c_void, text: *const c_char, thinking: bool);

#[repr(C)]
#[derive(Default)]
pub struct SkylaStats {
    pub prompt_tokens: u32,
    pub prefill_ms: f64,
    pub tokens: u32,
    pub tokens_per_second: f64,
    pub position: u32,
    pub context: u32,
}

pub struct Skyla {
    session: ManuallyDrop<Session<'static>>,
    tok: *mut Tokenizer,
    gguf: *mut Gguf,
    cancel: AtomicBool,
    name: CString,
}

fn error_out(out: *mut *mut c_char, message: String) {
    if !out.is_null() {
        let text = CString::new(message.replace('\0', " ")).unwrap_or_default();
        unsafe { *out = text.into_raw() };
    }
}

unsafe fn text<'a>(p: *const c_char) -> &'a str {
    match p.is_null() {
        true => "",
        false => CStr::from_ptr(p).to_str().unwrap_or(""),
    }
}

fn open(path: &str, context: usize) -> anyhow::Result<Skyla> {
    let gguf = Box::into_raw(Box::new(Gguf::open(path)?));
    let tok = match Tokenizer::from_gguf(unsafe { &*gguf }) {
        Ok(t) => Box::into_raw(Box::new(t)),
        Err(e) => {
            drop(unsafe { Box::from_raw(gguf) });
            return Err(e);
        }
    };
    let session = Session::new(
        unsafe { &*gguf },
        unsafe { &*tok },
        context,
        KvPrecision::F16KeysQ8Values,
    );
    match session {
        Ok(session) => {
            let name = CString::new(session.info().model.replace('\0', " ")).unwrap_or_default();
            Ok(Skyla {
                session: ManuallyDrop::new(session),
                tok,
                gguf,
                cancel: AtomicBool::new(false),
                name,
            })
        }
        Err(e) => {
            unsafe {
                drop(Box::from_raw(tok));
                drop(Box::from_raw(gguf));
            }
            Err(e)
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn skyla_open(path: *const c_char, context: u32, error: *mut *mut c_char) -> *mut Skyla {
    match open(text(path), context as usize) {
        Ok(s) => Box::into_raw(Box::new(s)),
        Err(e) => {
            error_out(error, format!("{e:#}"));
            ptr::null_mut()
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn skyla_model_name(s: *const Skyla) -> *const c_char {
    (*s).name.as_ptr()
}

#[no_mangle]
pub unsafe extern "C" fn skyla_device_name(s: *const Skyla) -> *mut c_char {
    let device = (*s).session.info().device;
    CString::new(device).unwrap_or_default().into_raw()
}

#[allow(clippy::too_many_arguments)]
#[no_mangle]
pub unsafe extern "C" fn skyla_chat(
    s: *mut Skyla,
    message: *const c_char,
    system: *const c_char,
    temperature: f32,
    top_p: f32,
    max_tokens: u32,
    callback: TokenCallback,
    user: *mut c_void,
    stats: *mut SkylaStats,
    error: *mut *mut c_char,
) -> bool {
    let s = &mut *s;
    s.cancel.store(false, Ordering::Relaxed);
    let settings = Settings {
        temperature,
        top_p,
        max_tokens: max_tokens as usize,
        ..Settings::default()
    };
    let mut sampler = Sampler::from_settings(&settings, seed_from_clock());
    let system = Some(text(system)).filter(|t| !t.trim().is_empty());
    let cancel = &s.cancel;
    let emit = |piece: Piece| {
        if let Piece::Text { text, thinking } = piece {
            let c = CString::new(text.replace('\0', "")).unwrap_or_default();
            callback(user, c.as_ptr(), thinking);
        }
        !cancel.load(Ordering::Relaxed)
    };
    match s
        .session
        .turn(text(message), system, &mut sampler, settings.max_tokens, emit)
    {
        Ok(st) => {
            if !stats.is_null() {
                *stats = SkylaStats {
                    prompt_tokens: st.prompt_tokens as u32,
                    prefill_ms: st.prefill_ms,
                    tokens: st.tokens as u32,
                    tokens_per_second: st.tokens_per_second,
                    position: st.position as u32,
                    context: st.context as u32,
                };
            }
            true
        }
        Err(e) => {
            error_out(error, format!("{e:#}"));
            false
        }
    }
}

#[no_mangle]
pub unsafe extern "C" fn skyla_cancel(s: *const Skyla) {
    (*s).cancel.store(true, Ordering::Relaxed);
}

#[no_mangle]
pub unsafe extern "C" fn skyla_reset(s: *mut Skyla) {
    (*s).session.reset();
}

#[no_mangle]
pub unsafe extern "C" fn skyla_free(s: *mut Skyla) {
    if s.is_null() {
        return;
    }
    let mut s = Box::from_raw(s);
    ManuallyDrop::drop(&mut s.session);
    drop(Box::from_raw(s.tok));
    drop(Box::from_raw(s.gguf));
}

#[no_mangle]
pub unsafe extern "C" fn skyla_string_free(p: *mut c_char) {
    if !p.is_null() {
        drop(CString::from_raw(p));
    }
}
