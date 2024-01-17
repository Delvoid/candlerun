use candle_transformers::models::mistral::Model as Mistral;
use candle_transformers::models::quantized_llama::ModelWeights;

pub enum ModelType {
    Mistral(Mistral),
    Quantized(ModelWeights),
}
