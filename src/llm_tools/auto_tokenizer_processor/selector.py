from llm_tools.auto_tokenizer_processor.processor_wrapper import ProcessorWrapper
from llm_tools.auto_tokenizer_processor.tokenizer_wrapper import TokenizerWrapper
from llm_tools.config.config import Config
from llm_tools.type.model_type import ModelType


def select_tokenizer_processor(config: Config):
    is_vision_model = ModelType.it_vision_models(config.llm_model.llm_model_type)

    if is_vision_model:
        return ProcessorWrapper(config, None)

    return TokenizerWrapper(config.llm_model, None)
