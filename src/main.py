import os
from pprint import pprint

import tyro
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from pydantic import BaseModel, DirectoryPath, Field, FilePath

from cache import Cache
from logger import get_logger
from models import extract_prompt_en
from rule import Rule, generate_robust_rule
from utils import process_dataset, read_dataset

load_dotenv()

logger = get_logger(__name__)


class Args(BaseModel):
    data_folder: DirectoryPath = Field(
        # default=DirectoryPath("data"),
        description="Path to the data folder."
    )
    dataset_filename: str = Field(
        # default="dataset.json",
        description="Name of the dataset JSON file within the data folder.",
    )
    max_attempts: int = Field(
        # default=5,
        ge=1,
        le=10,
        description="Maximum number of attempts.",
    )
    cache_filename: str | None = Field(
        None,
        description="Name of the cache file within the data folder.",
    )
    # timeout: float = Field(
    #     # default=30.0,
    #     ge=0.0,
    #     description="Timeout duration in seconds.",
    # )


model = init_chat_model(
    "gemini-2.5-flash",
    model_provider="google_genai",
    api_key=os.getenv("GEMINI_API_KEY"),
)


def main(args: Args):
    # print(args)

    dataset = read_dataset(filename=args.dataset_filename, data_folder=args.data_folder)

    processed_dataset = process_dataset(dataset=dataset, data_folder=args.data_folder)

    # pprint("Original dataset:")
    # pprint(dataset[0])

    pprint("Processed dataset:")
    pprint(processed_dataset[0])

    if args.cache_filename:
        logger.info(f"Loading cache from {args.cache_filename}...")
        if args.cache_filename.endswith(".pkl"):
            my_cache = Cache.load_from_file_pickle(
                filepath=os.path.join(args.data_folder, args.cache_filename)
            )
        elif args.cache_filename.endswith(".json"):
            my_cache = Cache.load_from_file_json(
                filepath=os.path.join(args.data_folder, args.cache_filename)
            )
        else:
            raise ValueError("Unsupported cache file format.")
    else:
        logger.info("No cache file specified, using empty cache.")
        my_cache = Cache()

        # (
        #     {
        #         "label": "carteira_oab",
        #         "extraction_schema": {
        #             "nome": "Nome do profissional, normalmente no canto superior esquerdo da imagem",
        #             "inscricao": "Número de inscrição do profissional",
        #             "seccional": "Seccional do profissional",
        #             "subsecao": "Subseção à qual o profissional faz parte",
        #             "categoria": "Categoria, pode ser ADVOGADO, ADVOGADA, SUPLEMENTAR, ESTAGIARIO, ESTAGIARIA",
        #             "endereco_profissional": "Endereço do profissional",
        #             "situacao": "Situação do profissional, normalmente no canto inferior direito.",
        #         },
        #         "pdf_path": "oab_2.pdf",
        #     },
        # )

        # # Initialize Mother Cache
        # dict_cache = {}
        for data in processed_dataset[:1]:
            #     label = data["label"]
            #     pdf_text = data["pdf_text"]

            #     # Initialize for each label
            #     if label not in dict_cache:
            #         dict_cache[label] = {}

            #     # Initialize for each field within the label a Cache
            #     for key in data["extraction_schema"].keys():
            #         if key not in dict_cache[label]:
            #             dict_cache[label][key] = Cache()

            #     # For each field, in each label, test all the rules in the cache sequentially
            #     for field in data["extraction_schema"].keys():
            #         print(f"Processing field '{field}' for label '{label}'")
            #         cache = dict_cache[label][field]
            #         for rule in cache:
            #             print("Testing rule:", rule)

            #             extracted_value = rule.apply(pdf_text)
            #             if rule.validate(extracted_value):
            #                 cache.add_hit(rule)
            #             break

            # def print_dict_cache():
            #     for label, fields in dict_cache.items():
            #         print(f"Label: {label}")
            #         for field, cache in fields.items():
            #             print(f"  Field: {field}, Cache length: {cache.length}")

            # print_dict_cache()

            agent = create_agent(
                # model=model, tools=[], response_format=OABSchema, checkpointer=memory
                model=model,
                tools=[],
                response_format=data["pydantic_model"],
            )

            response = agent.invoke(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": extract_prompt_en.format(
                                text=data["pdf_text"],
                                schema=data["pydantic_model"],
                            ),
                        }
                    ]
                },
                # config=config,
            )

            # print(response["structured_response"])

            model_dict = response["structured_response"].model_dump()
            # normalized_dict = {k: clean_llm_output(v) for k, v in model_dict.items()}
            normalized_dict = {k: v for k, v in model_dict.items()}
            response["structured_response"] = data["pydantic_model"](**normalized_dict)

            # print(response["structured_response"])

            agent_rule = create_agent(
                # model=model, tools=[], response_format=Rule, checkpointer=memory
                model=model,
                tools=[],
                response_format=Rule,
            )

            pdf_text = data["pdf_text"]
            llm_response = response["structured_response"].model_dump()

            generated_rules = {}

            for field, value in llm_response.items():
                # print("Generating rule for field:", field, "with value:", value)
                if value is not None:
                    # Preparar os inputs para o NOVO prompt
                    field_name = field
                    field_value = value
                    field_description = data["extraction_schema"][field]

                    rule_object = generate_robust_rule(
                        agent_rule,
                        pdf_text,
                        field_name,
                        field_value,
                        field_description,
                        max_attempts=5,
                    )

                    generated_rules[field] = rule_object
                    # print("Generated rule for field", field, ":", generated_rules[field])
                break

            logger.info("All generated rules:")
            pprint(generated_rules)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
