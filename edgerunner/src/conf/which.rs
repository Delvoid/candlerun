use clap::ValueEnum;
use serde::{Deserialize, Serialize};

/// Enum representing available models.
#[derive(Clone, Debug, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub enum Which {
    #[value(name = "7b-mistral")]
    Mistral7b,
    #[value(name = "7b-mistral-instruct")]
    Mistral7bInstruct,
    #[value(name = "7b-mistral-instruct-Q2")]
    Mistral7bInstructQ2,
    #[value(name = "7b-zephyr-b")]
    Zephyr7bBeta,
    #[value(name = "mixtral")]
    Mixtral,
    #[value(name = "mixtral-instruct")]
    MixtralInstruct,
    #[value(name = "openchat-3.5")]
    OpenChat35,
}

impl Which {
    pub fn is_mistral(&self) -> bool {
        match self {
            Self::OpenChat35
            | Self::Zephyr7bBeta
            | Self::Mixtral
            | Self::MixtralInstruct
            | Self::Mistral7bInstructQ2
            | Self::Mistral7b
            | Self::Mistral7bInstruct => true,
        }
    }

    /// Determines if the model is of type Zephyr
    pub fn is_zephyr(&self) -> bool {
        matches!(self, Self::Zephyr7bBeta)
    }

    /// Determines if the model is OpenChat
    pub fn is_open_chat(&self) -> bool {
        matches!(self, Self::OpenChat35)
    }

    // Returns the repository associated with the model
    pub fn tokenizer_repo(&self) -> &'static str {
        match self {
            Self::Mixtral | Self::MixtralInstruct => "mistralai/Mixtral-8x7B-v0.1",
            Self::Mistral7b
            | Self::Mistral7bInstruct
            | Self::Mistral7bInstructQ2
            | Self::Zephyr7bBeta => "mistralai/Mistral-7B-v0.1",
            Self::OpenChat35 => "openchat/openchat_3.5",
        }
    }

    pub fn get_repo_and_filename(&self) -> (&'static str, &'static str) {
        match self {
            Self::Mixtral => (
                "TheBloke/Mixtral-8x7B-v0.1-GGUF",
                "mixtral-8x7b-v0.1.Q4_K_M.gguf",
            ),
            Self::MixtralInstruct => (
                "TheBloke/Mixtral-8x7B-Instruct-v0.1-GGUF",
                "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf",
            ),
            Self::Mistral7b => (
                "TheBloke/Mistral-7B-v0.1-GGUF",
                "mistral-7b-v0.1.Q4_K_S.gguf",
            ),
            Self::Mistral7bInstruct => (
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                "mistral-7b-instruct-v0.1.Q4_K_S.gguf",
            ),

            Self::Mistral7bInstructQ2 => (
                "TheBloke/Mistral-7B-Instruct-v0.1-GGUF",
                "mistral-7b-instruct-v0.1.Q2_K.gguf",
            ),

            Self::Zephyr7bBeta => ("TheBloke/zephyr-7B-beta-GGUF", "zephyr-7b-beta.Q4_K_M.gguf"),
            Self::OpenChat35 => ("TheBloke/openchat_3.5-GGUF", "openchat_3.5.Q4_K_M.gguf"),
        }
    }
}
