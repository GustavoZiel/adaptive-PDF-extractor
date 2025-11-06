import json
import os
import random
import string

import tyro
from faker import Faker
from pydantic import BaseModel, Field

from src.logger import get_logger

logger = get_logger(__name__, level="DEBUG")
fake = Faker("pt_BR")


LABEL = "carteira_oab"
EXTRACTION_SCHEMA = {
    "nome": "Nome do profissional, normalmente no canto superior esquerdo da imagem",
    "inscricao": "Número de inscrição do profissional",
    "seccional": "Seccional do profissional",
    "subsecao": "Subseção à qual o profissional faz parte",
    "categoria": "Categoria, pode ser ADVOGADO, ADVOGADA, SUPLEMENTAR, ESTAGIARIO, ESTAGIARIA",
    "endereco_profissional": "Endereço do profissional",
    "telefone_profissional": "Telefone do profissional",
    "situacao": "Situação do profissional, normalmente no canto inferior direito.",
}


class Args(BaseModel):
    save_path: str = Field(..., description="Path to the data folder.")
    num_samples: int = Field(
        ..., description="Number of samples to generate for the dataset."
    )
    filename: str | None = Field(
        None,
        description="Filename for the generated dataset. If None, generates name based on num_samples and seed.",
    )
    seed: int = Field(1, description="Random seed for reproducibility.")


def seed_random(seed_value):
    random.seed(seed_value)
    fake.seed_instance(seed_value)


def generate_canonical_record():
    return {
        "nome": fake.name(),
        "inscricao": fake.rg(),
        "seccional": fake.state_abbr(),
        "subsecao": f"{fake.city()} - {fake.state()}",
        "categoria": random.choice(
            ["ADVOGADO", "ADVOGADA", "SUPLEMENTAR", "ESTAGIARIO"]
        ),
        "endereco_profissional": fake.address().replace("\n", ", "),
        "telefone_profissional": fake.phone_number(),
        "situacao": random.choice(["Situação Regular", "Situação Irregular"]),
    }


def generate_wrong_data(field, canonical_value):
    wrong_data = canonical_value

    while wrong_data == canonical_value:
        if field == "nome":
            wrong_data = (
                canonical_value
                + " "
                + "".join(random.choices(string.ascii_letters + string.digits, k=3))
            )
        elif field == "inscricao":
            wrong_data = str(random.randint(0, 99))
        elif field == "seccional":
            wrong_data = "".join(
                random.choices(
                    string.ascii_letters + string.digits, k=random.randint(3, 5)
                )
            )
        elif field == "subsecao":
            wrong_data = f"{fake.city()} {random.randint(1, 99)} - {fake.state()}"
        elif field == "categoria":
            wrong_data = random.choice(["ADV", "EST", "SUP", "ADVOGADA123", ""])
        elif field == "endereco_profissional":
            wrong_data = (
                canonical_value.replace(",", ";") + " #" + str(random.randint(1, 100))
            )
        elif field == "telefone_profissional":
            wrong_data = "".join(
                random.choices(string.digits + string.ascii_letters, k=10)
            )
        elif field == "situacao":
            wrong_data = random.choice(["SUSPENSO", "CANCELADO", "BLOQUEADO", ""])
        else:
            wrong_data = "DADO ERRADO " + str(random.randint(0, 999))

    return wrong_data


def fuzz_text(text):
    if not text:
        return text
    text = str(text)
    noise_map = {
        "O": "0",
        "o": "0",
        "l": "1",
        "i": "1",
        "S": "5",
        "s": "5",
        "B": "8",
        "A": "4",
    }
    fuzzed = ""
    for char in text:
        # if char in noise_map and random.random() < 0.15:
        #     fuzzed += noise_map[char]
        # elif (
        if char == " " and random.random() < 0.1:
            pass
        else:
            fuzzed += char

    return fuzzed


def generate_sample():
    canonical_record = generate_canonical_record()

    ocr_chunks = []
    expected_answer = {}

    # if random.random() < 0.3:
    #     ocr_chunks.append(
    #         random.choice(["ORDEM DOS ADV0GAD0S", "SCAN DOC 001", fake.company()])
    #     )

    for field in EXTRACTION_SCHEMA.keys():
        state = random.choice(
            ["correct"] * 7 + ["omitted"] * 3
            # ["correct"] * 5 + ["wrong"] * 2 + ["omitted"] * 3 + ["placeholder"] * 0
        )

        value = None
        expected_value = None

        if state == "correct":
            expected_value = canonical_record[field]
            value = expected_value
        elif state == "omitted":
            expected_value = None
            expected_answer[field] = expected_value
            ocr_chunks.append((field.replace("_", " ").title(), expected_value))
            continue

        #     elif state == "wrong":
        #         value = generate_wrong_data(field, canonical_record[field])
        #         expected_value = value
        #     # elif state == "placeholder":
        #     #     value = random.choice(["N/D", "---", "???", "NA", "ILEGIVEL"])
        #     #     expected_value = value  # O modelo deve ser capaz de extrair "N/D"

        expected_answer[field] = expected_value

        label_text_base = field.replace("_", " ").title()

        label_state = random.choice(
            ["full"] * 9 + ["omitted"] * 1
            # ["full"] * 5 + ["mixed"] * 3 + ["partial"] * 1 + ["omitted"] * 1
        )

        # fuzzed_value = fuzz_text(value)
        fuzzed_value = value

        if label_state == "full":
            ocr_chunks.append((label_text_base, fuzzed_value))
        elif label_state == "omitted":
            ocr_chunks.append((None, fuzzed_value))

        #     elif label_state == "partial":
        #         ocr_chunks.append(fuzz_text(label_text_base.split()[0]))
        #         ocr_chunks.append(fuzzed_value)

        # elif label_state == "mixed":
        #     if random.random() < 0.5:
        #         ocr_chunks.append(fuzz_text(f"{label_text_base.split()[0]} {value}"))
        #     else:
        #         ocr_chunks.append(fuzz_text(f"{label_text_base.split()[0]}{value}"))

    if random.random() < 0.33:
        random.shuffle(ocr_chunks)

    logger.debug(f"ocr_chunks: {ocr_chunks}")
    logger.debug(f"expected_answer: {expected_answer}")

    final_ocr_text = ""
    for chunk in ocr_chunks:
        label, value = chunk

        if label is not None:
            final_ocr_text += label

        separator = random.choice(
            ["\n"] * 8 + [""] * 2 + [" "] * 6 + ["   "] * 2 + ["\t"] * 2,
        )
        final_ocr_text += separator

        if value is not None:
            # final_ocr_text += f" {value}"
            final_ocr_text += value

    logger.debug("final_ocr_text")
    print(final_ocr_text)

    return {
        "label": LABEL,
        "extraction_schema": EXTRACTION_SCHEMA,
        "pdf_text": final_ocr_text,
        "expected_answer": expected_answer,
    }


def write_json(data, filename, path):
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    logger.info(f"Dataset gerado e salvo em: {output_path}")


def main(args: Args):
    logger.info("Iniciando a geração do dataset falso...")

    seed_random(args.seed)

    dataset = [generate_sample() for _ in range(args.num_samples)]

    if not args.filename:
        filename = f"fake_dataset_{args.num_samples}samples_seed_{args.seed}.json"
    else:
        filename = args.filename

    write_json(dataset, filename, args.save_path)

    logger.info("Geração do dataset falso concluída.")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
