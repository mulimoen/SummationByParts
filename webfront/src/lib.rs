use wasm_bindgen::prelude::*;

mod euler;
pub use crate::euler::EulerUniverse;

mod maxwell;
pub use crate::maxwell::MaxwellUniverse;

mod shallow_water;
pub use crate::shallow_water::ShallowWaterUniverse;

#[wasm_bindgen]
pub fn set_panic_hook() {
    console_error_panic_hook::set_once();
}

#[wasm_bindgen]
pub fn set_console_logger() {
    use std::sync::Once;
    static LOG_INIT: Once = Once::new();

    LOG_INIT.call_once(|| console_log::init_with_level(log::Level::Trace).unwrap());
}
