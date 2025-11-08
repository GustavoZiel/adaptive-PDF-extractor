"""Generate synthetic dataset for testing adaptive PDF extraction.

This script creates fake OAB (Brazilian Bar Association) card data with:
- Realistic Brazilian names, addresses, phone numbers
- Random field omissions to simulate OCR errors
- Configurable sample size and random seed for reproducibility

Used for testing the extraction pipeline without real sensitive data.
"""

import json
import os
import random
import string

import tyro
from faker import Faker
from pydantic import BaseModel, Field

from src.logger import get_logger

logger = None
fake = Faker("pt_BR")  # Brazilian Portuguese locale


# ============================================================================
# SECTION 1: Data Schema Configuration
# ============================================================================

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
    """CLI arguments for fake data generation."""

    save_path: str = Field(..., description="Path to the data folder.")
    num_samples: int = Field(
        ..., description="Number of samples to generate for the dataset."
    )
    dataset_filename: str | None = Field(
        None,
        description="Filename for the generated dataset. If None, generates name based on num_samples and seed.",
    )
    seed: int = Field(1, description="Random seed for reproducibility.")
    log_level: str = Field(
        "INFO", description="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)."
    )


# ============================================================================
# SECTION 2: Random Seed Management
# ============================================================================


def seed_random(seed_value: int):
    """Set random seed for reproducibility.

    Args:
        seed_value: Seed value for random number generators
    """
    random.seed(seed_value)
    fake.seed_instance(seed_value)
    logger.debug("Random seed set to: %d", seed_value)


# ============================================================================
# SECTION 3: Data Generation
# ============================================================================


def generate_canonical_record() -> dict:
    """Generate a canonical OAB card record with valid Brazilian data.

    Uses Faker library to generate realistic Brazilian names, addresses,
    phone numbers, and other personal information.

    Returns:
        Dictionary with all OAB card fields populated with valid data
    """
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


def generate_wrong_data(field: str, canonical_value: str) -> str:
    """Generate intentionally wrong data for a field (currently unused).

    This function can be used to simulate OCR errors or corrupted data
    by generating values that don't match the expected format.

    Args:
        field: Field name to generate wrong data for
        canonical_value: Original correct value

    Returns:
        Incorrect value different from canonical_value
    """
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


# ============================================================================
# SECTION 4: OCR Noise Simulation
# ============================================================================


def fuzz_text(text: str) -> str:
    """Apply OCR-like noise to text (currently minimal).

    Simulates OCR errors by occasionally removing spaces.
    Can be extended with character substitutions (O→0, l→1, etc.).

    Args:
        text: Input text to fuzz

    Returns:
        Text with simulated OCR noise applied
    """
    if not text:
        return text

    text = str(text)

    # Character substitution map (currently commented out in original)
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
        # Random space removal (10% chance)
        if char == " " and random.random() < 0.1:
            pass  # Skip space
        else:
            fuzzed += char

    return fuzzed


# ============================================================================
# SECTION 5: Sample Generation
# ============================================================================


def generate_sample() -> dict:
    """Generate a single fake OAB card sample.

    Creates a sample with:
    - 70% fields present (correct)
    - 30% fields omitted (simulating OCR failures)
    - Random label presence/omission
    - Random field ordering (33% shuffled)
    - Varied separator styles (newline, space, tab)

    Returns:
        Dictionary with label, extraction_schema, pdf_text, expected_answer
    """
    canonical_record = generate_canonical_record()

    ocr_chunks = []
    expected_answer = {}

    # Generate each field with random state
    for field in EXTRACTION_SCHEMA.keys():
        # 70% correct, 30% omitted
        state = random.choice(["correct"] * 7 + ["omitted"] * 3)

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

        expected_answer[field] = expected_value

        # Generate field label
        label_text_base = field.replace("_", " ").title()

        # 90% with label, 10% without label
        label_state = "full"  # Only "full" labels for now
        # label_state = random.choice(["full"] * 9 + ["omitted"] * 1)

        fuzzed_value = value  # Can apply fuzz_text(value) for noise

        if label_state == "full":
            ocr_chunks.append((label_text_base, fuzzed_value))
        elif label_state == "omitted":
            ocr_chunks.append((None, fuzzed_value))

    # Shuffle chunks 33% of the time (simulate unordered OCR)
    if random.random() < 0.33:
        random.shuffle(ocr_chunks)

    logger.debug("OCR chunks: %s", ocr_chunks)
    logger.debug("Expected answer: %s", expected_answer)

    # Assemble final OCR text
    final_ocr_text = ""
    for chunk in ocr_chunks:
        label, value = chunk

        # Add label if present
        if label is not None:
            final_ocr_text += label

        # Random separator: 40% newline, 30% space, 10% nothing, etc.
        separator = random.choice(
            ["\n"] * 8 + [""] * 2 + [" "] * 6 + ["   "] * 2 + ["\t"] * 2,
        )
        final_ocr_text += separator

        # Add value if present
        if value is not None:
            final_ocr_text += value

    logger.debug("Final OCR text:\n%s", final_ocr_text)

    return {
        "label": LABEL,
        "extraction_schema": EXTRACTION_SCHEMA,
        "pdf_text": final_ocr_text,
        "expected_answer": expected_answer,
    }


# ============================================================================
# SECTION 6: File Output
# ============================================================================


def write_json(data: list, filename: str, path: str):
    """Write dataset to JSON file.

    Args:
        data: List of sample dictionaries to save
        filename: Output filename
        path: Directory path to save to
    """
    os.makedirs(path, exist_ok=True)
    output_path = os.path.join(path, filename)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    logger.info("Dataset saved to: %s", output_path)


# ============================================================================
# SECTION 7: Main Entry Point
# ============================================================================


def main(args: Args):
    """Main function to generate fake dataset.

    Args:
        args: CLI arguments with save_path, num_samples, filename, seed
    """
    global logger
    logger = get_logger(__name__, level=args.log_level)

    logger.info("Starting fake dataset generation...")

    # Set random seed for reproducibility
    seed_random(args.seed)

    # Generate samples
    logger.info("Generating %d samples...", args.num_samples)
    dataset = [generate_sample() for _ in range(args.num_samples)]

    # Determine output filename
    if not args.dataset_filename:
        dataset_filename = (
            f"fake_dataset_{args.num_samples}samples_seed_{args.seed}.json"
        )
    else:
        dataset_filename = (
            args.dataset_filename
            if args.dataset_filename.endswith(".json")
            else args.dataset_filename + ".json"
        )

    # Save to disk
    write_json(dataset, dataset_filename, args.save_path)

    logger.info("✅ Fake dataset generation complete!")
    logger.info("Generated %d samples with seed %d", args.num_samples, args.seed)


if __name__ == "__main__":
    args = tyro.cli(Args)
    main(args)
