import json
import os
import random
import string

import tyro
from faker import Faker
from pydantic import BaseModel, Field

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
        "situacao": "Regular",
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

    if random.random() < 0.3:
        ocr_chunks.append(
            random.choice(["ORDEM DOS ADV0GAD0S", "SCAN DOC 001", fake.company()])
        )

    for field in EXTRACTION_SCHEMA.keys():
        state = random.choice(
            [
                "correct",
                "correct",
                "correct",
                "correct",
                "correct",
                "wrong",
                "wrong",
                "omitted",
                "omitted",
                "omitted",
            ]
        )  # , "placeholder"])

        value = None
        expected_value = None

        if state == "correct":
            value = canonical_record[field]
            expected_value = value
        elif state == "wrong":
            value = generate_wrong_data(field, canonical_record[field])
            expected_value = value
        # elif state == "placeholder":
        #     value = random.choice(["N/D", "---", "???", "NA", "ILEGIVEL"])
        #     expected_value = value  # O modelo deve ser capaz de extrair "N/D"
        elif state == "omitted":
            expected_value = None
            # # Mesmo omitido, o rótulo pode aparecer
            # if random.random() < 0.3:
            #     ocr_chunks.append(fuzz_text(field.replace("_", " ").title()))
            continue

        expected_answer[field] = expected_value

        label_text_base = field.replace("_", " ").title()
        label_state = random.choice(
            [
                "full",
                "full",
                "full",
                "full",
                "full",
                "mixed",
                "mixed",
                "mixed",
                "partial",
                "omitted",
            ]
        )

        fuzzed_value = fuzz_text(value)

        if label_state == "full":
            ocr_chunks.append(fuzz_text(label_text_base))
            ocr_chunks.append(fuzzed_value)

        elif label_state == "partial":
            ocr_chunks.append(fuzz_text(label_text_base.split()[0]))
            ocr_chunks.append(fuzzed_value)

        elif label_state == "omitted":
            ocr_chunks.append(fuzzed_value)

        elif label_state == "mixed":
            if random.random() < 0.5:
                ocr_chunks.append(fuzz_text(f"{label_text_base.split()[0]} {value}"))
            else:
                ocr_chunks.append(fuzz_text(f"{label_text_base.split()[0]}{value}"))

    final_ocr_text = ""
    for chunk in ocr_chunks:
        final_ocr_text += chunk
        separator = random.choice(
            [
                "\n",
                "\n",
                "\n",
                "\n",
                "\n",
                "\n",
                "\n",
                "\n",
                " ",
                " ",
                " ",
                "",
                "",
                "   ",
                "   ",
                "\t",
                "\t",
            ]
        )
        final_ocr_text += separator

    return {
        "label": LABEL,
        "extraction_schema": EXTRACTION_SCHEMA,
        "ocr_text": final_ocr_text,
        "expected_answer": expected_answer,
    }


class Args(BaseModel):
    save_path: str = Field(..., description="Path to the data folder.")
    num_samples: int = Field(
        ..., description="Number of samples to generate for the dataset."
    )
    seed: int = Field(42, description="Random seed for reproducibility.")


def main(args: Args):
    seed_random(args.seed)

    dataset = [generate_sample() for _ in range(args.num_samples)]

    os.makedirs(args.save_path, exist_ok=True)
    output_path = os.path.join(args.save_path, "fake_dataset.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)

    print(f"Dataset gerado e salvo em: {output_path}")


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
