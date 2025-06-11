# Rare Disease Classification using LoRA-Adapted LLaVA (PEFT)

This project fine-tunes a multimodal vision-language model (LLaVA) on the ODIR dataset using **LoRA** (Low-Rank Adaptation) for classifying rare ocular diseases from fundus images. The pipeline is built on top of [MedTrinity-25M](https://github.com/UCSC-VLAA/MedTrinity-25M) and supports medical prompt-based generation and structured evaluation.

---

## Dataset: ODIR

- 5,000 patient fundus images (left + right eyes)
- Real-world quality variation (Canon, Zeiss, Kowa cameras)
- 8 diagnostic classes:
  - Normal, Diabetes, Glaucoma, Cataract, Age-related Macular Degeneration, Hypertension, Pathological Myopia, Other Abnormalities

---

## Setup & Installation

1. **Clone this repo and create a virtual environment**
```bash
git clone https://github.com/yourusername/rare-disease-peft-llava.git
cd rare-disease-peft-llava
python3 -m venv venv
source venv/bin/activate

---

## Zero shot Inference, Adapting PEFT and Trained model inference
Make sure to setup LLaVA basemodel, ODIR dataset in your env before running all cells.

Open peft_for_llava.ipynb in Google Colab or Jupyter

Set runtime to GPU

Run through all setup, LoRA loading, and inference cells

Predictions will be saved in CSV format like llava_peft_test_results.csv

## LoRA configuration used
LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

#Base model inference prompts
Prompt 1 -> Analyze the provided fundus image and identify which of the following ocular conditions are present, if any.
Possible conditions:
- Normal (N)
- Diabetes (D)
- Glaucoma (G)
- Cataract (C)
- Age-related Macular Degeneration (A)
- Hypertension (H)
- Pathological Myopia (M)
- Other diseases or abnormalities (0)

List all conditions observed in the image. If no condition is present, list "N".

Prompt 2 -> Analyze the provided fundus image and identify which of the following ocular conditions are present, if any.
Possible conditions:
- Normal (N)
- Diabetes (D)
- Glaucoma (G)
- Cataract (C)
- Age-related Macular Degeneration (A)
- Hypertension (H)
- Pathological Myopia (M)
- Other diseases or abnormalities (0)

List all conditions observed in the image. If the eye is healthy, respond with: "Normal"!.
If multiple conditions are present, list all of them in a Python list, e.g., ["Diabetes", "Glaucoma"].

Prompt 3 -> Does this look like the fundus of a human eye?
Prompt 4 -> What can you identify from this image?
Prompt 5 -> What organ is this?