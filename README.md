# Project Progression and Milestones

## TODOs

- [ ] Clean the code
- [ ] See how to generate and share reports in wandb automatically.
- [ ] Improve logic with cache for user
- [ ] Export answers to CSV or JSON

- Fake Data Generation
  - Its assumed that the data will come already cleaned, in a sense that in forms the user wouldn't be able to write his name in the age field, for example. Therefore, what can be messy is the format of the data, missing values, etc. This messiness appear when reading the OCR text from images. This messy is what we aim to simulate.
  - The correctness of the model is determined by the first LLM call that extract from the OCR text the data in a structured format.
  - 