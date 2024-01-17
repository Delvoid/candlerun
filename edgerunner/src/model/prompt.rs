use crate::conf::which::Which;
use anyhow::Result;

#[allow(dead_code)]
const DEFAULT_PROMPT: &str = "My favorite theorem is ";
const DEFAULT_SYSTEM_PROMPT: &str = "<s>[INST] Always respond with concise messages with correct grammar. Avoid html tags, garbled content, and words that run into one another. If you don't know the answer to a question say 'I don't know'.[/INST] Understood! I will always respond with concise messages and correct grammar. If I don't know the answer to a question, I will say 'I don't know'.</s>";

#[derive(Debug)]
pub struct GeneratedPrompt(pub String);

impl GeneratedPrompt {
    // Method to access the internal string
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

#[derive(Debug)]
enum Prompt {
    Chat(String),
    #[allow(dead_code)]
    One(String),
}

impl Prompt {
    /// Generates the appropriate prompt string based on the prompt type and user input
    pub fn generate_prompt(
        &self,
        which: &Which,
        conversation_history: Option<&[String]>,
    ) -> anyhow::Result<GeneratedPrompt> {
        match self {
            Prompt::One(prompt) => Ok(GeneratedPrompt(prompt.clone())),
            Prompt::Chat(prompt) => {
                self.generate_user_input_prompt(which, prompt, conversation_history)
            }
        }
    }

    /// Generates a prompt for the user input based on the model type Alias = Type;
    fn generate_user_input_prompt(
        &self,
        which: &Which,
        prompt_text: &str,
        conversation_history: Option<&[String]>,
    ) -> Result<GeneratedPrompt> {
        let is_instruct = matches!(
            which,
            Which::Mistral7bInstruct | Which::MixtralInstruct | Which::Mistral7bInstructQ2
        );

        // Handling different model types for prompt formatting
        if which.is_open_chat() {
            Ok(GeneratedPrompt(format!(
                "GPT4 Correct User: {prompt_text}GPT4 Correct Assistant:"
            )))
        } else if which.is_zephyr() {
            Ok(GeneratedPrompt(format!(
                "<|user|>\n{prompt_text}</s>\n<|assistant|>"
            )))
        } else if which.is_mistral() {
            if is_instruct {
                self.generate_mistral_prompt(prompt_text, conversation_history)
            } else {
                Ok(GeneratedPrompt(format!("[INST] {prompt_text} [/INST]")))
            }
        } else {
            Ok(GeneratedPrompt(prompt_text.to_string()))
        }
    }

    /// Generates a prompt for the Mistral model with the conversation history
    fn generate_mistral_prompt(
        &self,
        text_from_chat: &str,
        conversation_history: Option<&[String]>,
    ) -> anyhow::Result<GeneratedPrompt> {
        let prompt = if let Some(history) = conversation_history {
            // Use the last two entries from the conversation history
            let history_split = history
                .iter()
                .rev()
                .take(2)
                .rev()
                .map(|x| x.to_string())
                .collect::<Vec<String>>();

            if history_split.is_empty() {
                format!(
                    "{}\n[INST] {} [/INST] ",
                    DEFAULT_SYSTEM_PROMPT, text_from_chat
                )
            } else {
                format!(
                    "{}\n[INST] {} [/INST] {}\n[INST] {} [/INST] ",
                    DEFAULT_SYSTEM_PROMPT, history_split[0], history_split[1], text_from_chat
                )
            }
        } else {
            // No conversation history provided, use a standard format
            format!(
                "{} [INST] {} [/INST]",
                DEFAULT_SYSTEM_PROMPT, text_from_chat
            )
        };

        Ok(GeneratedPrompt(prompt))
    }
}

pub fn handle_user_input(which: Which, input: &str) -> Result<GeneratedPrompt> {
    let prompt = Prompt::Chat(input.to_string());
    prompt.generate_prompt(&which, None)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Test for 'One' prompt type
    #[test]
    fn test_one_prompt() {
        let prompt = Prompt::One("Example prompt".to_string());
        let which = Which::Mistral7b; // Example model type
        let generated_prompt = prompt.generate_prompt(&which, None).unwrap();
        assert_eq!(generated_prompt.as_str(), "Example prompt");
    }

    // Test for 'Chat' prompt type with model type 'zephyr'
    #[test]
    fn test_chat_prompt_zephyr() {
        let prompt_text = "User question";
        let prompt = Prompt::Chat(prompt_text.to_string());
        let which = Which::Zephyr7bBeta; // Example model type
        let generated_prompt = prompt.generate_prompt(&which, None).unwrap();
        println!("Chat prompt (zephyr): {}", generated_prompt.as_str());
        assert!(generated_prompt.as_str().contains(prompt_text));
    }

    // Test for chat model type 'zephyr' formatting
    #[test]
    fn test_chat_prompt_zephyr_formatting() {
        let prompt_text = "User question";
        let prompt = Prompt::Chat(prompt_text.to_string());
        let which = Which::Zephyr7bBeta; // Example model type
        let generated_prompt = prompt.generate_prompt(&which, None).unwrap();
        println!("Chat prompt (zephyr): {}", generated_prompt.as_str());
        assert!(generated_prompt.as_str().contains("<|user|>"));
        assert!(generated_prompt.as_str().contains("</s>"));
        assert!(generated_prompt.as_str().contains("<|assistant|>"));
        assert_eq!(
            generated_prompt.as_str(),
            format!("<|user|>\n{}</s>\n<|assistant|>", prompt_text)
        );
    }

    // Test for 'Chat' prompt type with no conversation history
    #[test]
    fn test_chat_prompt_no_history() {
        let prompt_text = "User question";
        let prompt = Prompt::Chat(prompt_text.to_string());
        let which = Which::Mistral7bInstruct; // Example model type
        let generated_prompt = prompt.generate_prompt(&which, None).unwrap();
        println!("Chat prompt (no history): {}", generated_prompt.as_str());
        assert!(generated_prompt.as_str().contains(prompt_text));
        assert!(generated_prompt.as_str().contains(DEFAULT_SYSTEM_PROMPT));
    }

    // Test for 'Chat' prompt type with conversation history
    #[test]
    fn test_chat_prompt_with_history() {
        let prompt_text = "User question";
        let prompt = Prompt::Chat(prompt_text.to_string());
        let which = Which::Mistral7bInstruct; // Example model type
        let history = vec![
            "Previous user question".to_string(),
            "Previous bot response".to_string(),
        ];
        let generated_prompt = prompt.generate_prompt(&which, Some(&history)).unwrap();
        println!("Chat prompt (with history): {}", generated_prompt.as_str());
        assert!(generated_prompt.as_str().contains(prompt_text));
        assert!(generated_prompt.as_str().contains("Previous user question"));
        assert!(generated_prompt.as_str().contains("Previous bot response"));
        assert!(generated_prompt.as_str().contains(DEFAULT_SYSTEM_PROMPT));
    }

    // Test for chat model type 'mistral' formatting
    #[test]
    fn test_chat_prompt_mistral_formatting() {
        let prompt_text = "User question";
        let prompt = Prompt::Chat(prompt_text.to_string());
        let which = Which::Mistral7bInstruct; // Example model type
        let generated_prompt = prompt.generate_prompt(&which, None).unwrap();
        println!("Chat prompt (mistral): {}", generated_prompt.as_str());
        assert!(generated_prompt.as_str().contains("[INST]"));
        assert!(generated_prompt.as_str().contains("[/INST]"));
        assert_eq!(
            generated_prompt.as_str(),
            format!("{} [INST] {} [/INST]", DEFAULT_SYSTEM_PROMPT, prompt_text)
        );
    }

    // Test for chat model type 'mistral' formatting with conversation history
    #[test]
    fn test_chat_prompt_mistral_formatting_with_history() {
        let prompt_text = "User question";
        let prompt = Prompt::Chat(prompt_text.to_string());
        let which = Which::Mistral7bInstruct; // Example model type
        let history = vec![
            "Previous user question".to_string(),
            "Previous bot response".to_string(),
        ];
        let generated_prompt = prompt.generate_prompt(&which, Some(&history)).unwrap();
        println!("Chat prompt (mistral): {}", generated_prompt.as_str());
        assert!(generated_prompt.as_str().contains("[INST]"));
        assert!(generated_prompt.as_str().contains("[/INST]"));
        assert_eq!(
            generated_prompt.as_str(),
            format!(
                "{}\n[INST] {} [/INST] {}\n[INST] {} [/INST] ",
                DEFAULT_SYSTEM_PROMPT, history[0], history[1], prompt_text
            )
        );
    }

    // Test for chat model type 'mistral' formatting without conversation history
    #[test]
    fn test_chat_prompt_mistral_formatting_without_history() {
        let prompt_text = "User question";
        let prompt = Prompt::Chat(prompt_text.to_string());
        let which = Which::Mistral7bInstruct; // Example model type
        let generated_prompt = prompt.generate_prompt(&which, None).unwrap();
        println!("Chat prompt (mistral): {}", generated_prompt.as_str());
        assert!(generated_prompt.as_str().contains("[INST]"));
        assert!(generated_prompt.as_str().contains("[/INST]"));
        assert_eq!(
            generated_prompt.as_str(),
            format!("{} [INST] {} [/INST]", DEFAULT_SYSTEM_PROMPT, prompt_text)
        );
    }
}
