import os
import json
import pandas as pd
import google.generativeai as genai
import regex as re
from collections import defaultdict

# Setup
WORK_PATH = os.environ.get(
    "INTENTIONAL_COMPUTING_HOME",
    "/Users/choijuheonjuheon/DEV/KAIST_AI/Intention/cowork_place",
)
print(f"Working directory: {WORK_PATH}")

# Load intentions from CSV
intentions_csv = os.path.join(WORK_PATH, "experiments/config/intentions_detail.csv")
if not os.path.exists(intentions_csv):
    print(f"Error: {intentions_csv} not found")
    exit(1)

df = pd.read_csv(intentions_csv)
stated_intentions = {}
for _, row in df.iterrows():
    stated_intentions[row["trajectory_id"]] = row["intention"]

print(f"Loaded {len(stated_intentions)} intentions")

# Setup Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])
model = genai.GenerativeModel("gemini-2.0-flash")

generation_config = genai.GenerationConfig(
    temperature=0.1,
    top_p=1.0,
    top_k=32,
    max_output_tokens=256,
)

# Test API connection
print("\nTesting API connection...")
response = model.generate_content("What is 2+2?", generation_config=generation_config)
print(f"API Test Response: {response.text}")

# Dialogue template for clarification questions
dialogue_template_template = """
You are a helpful assistant that engages in a multi-turn conversation to better understand the user's intention.

[Input]
The user stated: "{stated_intention}"

[Goal]
Ask a clarification question to clarify their intention. 
The questions can be about the specifications of the target item, if the user is planning to shop, or a specific location, if the user is planning a tour.
Also, the questions can be about the specifications of the tools that the user wants to use, such as Slack or e-commerce websites.
When asking the question, guide the user with examples so that the user can understand your question clearly and answer easily.
Please ask diverse aspects of the intention (such as on targets or tools, as well as other related sub-tasks that the user may perform).
Ask only 2 clarification questions maximum.

[Context]
The Context provides you with information on whether previous questions and answers exist.

First_QA: {first_question_and_answer}
Second_QA: {second_question_and_answer}

[Output]
Only provide your question in a single sentence (at most 10 words).
"""

# Augmentation prompt template with clarification
augment_prompt_template = """You are an assistant that expands simple activity descriptions into diverse alternatives.

[Guideline]
Given a simple activity like "Find a jacket for men", generate 10 variations of the activity description.

Rules:
- Use the clarification questions you had with the user.
- The output must be in valid JSON format (keys: "1" to "10").
- Keep the sentence concise (at most 9 words).

Output structure:
- Variations 1–3: Broader or more generalized expressions of the activity.
- Variations 4–6: Slightly more specific or rephrased versions, using only the original information.
- Variations 7–10: Realistic user actions likely performed when carrying out the activity (e.g., read reviews, search on shopping apps).

[Example]
Activity: "Find a jacket for men"

```json
{
    "1": "Shopping",
    "2": "Online shopping",
    "3": "Browse for clothing",
    "4": "Search for jackets",
    "5": "Look up men's jackets in an online store",
    "6": "Navigate a shopping app to find a jacket",
    "7": "Use a search engine to locate jackets",
    "8": "Read customer reviews on shopping sites",
    "9": "Compare jacket prices across online stores",
    "10": "Watch review videos of jackets on YouTube"
}
```

[Clarification questions]
Below is a list of clarification questions. 
This provides a hint on the specific behaviors of the user, so augment the intention based on the information.
For example, if the user clarified that the user will buy a jacket at a specific website, you can include it (like, to 8-10). 

{clarification_questions_and_answers}

[Input]
Activity: "{stated_intention}"
"""

# Load existing clarifications if file exists
output_file = os.path.join(
    WORK_PATH, "experiments/asset/clarify_stated_intentions.json"
)

if os.path.exists(output_file):
    print(f"\nLoading existing clarifications: {output_file}")
    with open(output_file, "r") as json_file:
        final_output = json.load(json_file)
    print(f"Loaded {len(final_output)} existing clarifications")
else:
    print(f"\nFile not found, creating new: {output_file}")
    final_output = {}

# Check for missing tasks
missing_tasks = [
    task_id for task_id in stated_intentions.keys() if task_id not in final_output
]

if not missing_tasks:
    print("✅ All tasks already have clarifications!")
    print(f"Total tasks: {len(stated_intentions)}")
    print(f"Existing clarifications: {len(final_output)}")
    exit(0)

print(f"\nFound {len(missing_tasks)} tasks without clarifications:")
for task_id in missing_tasks:
    print(f"  - {task_id}: {stated_intentions[task_id]}")

print(f"\n🤖 Starting interactive clarification process...")
print("=" * 60)

for task_id in missing_tasks:
    stated_intention = stated_intentions[task_id]

    print(f"\n📋 Task: {task_id}")
    print(f"💭 Intention: {stated_intention}")
    print("-" * 40)

    # Clarification QAs
    qa_pairs = []
    for turn in range(2):
        first_qa = qa_pairs[0] if len(qa_pairs) > 0 else ("", "")
        second_qa = qa_pairs[1] if len(qa_pairs) > 1 else ("", "")

        prompt = (
            dialogue_template_template.replace("{stated_intention}", stated_intention)
            .replace(
                "{first_question_and_answer}",
                f"Q: {first_qa[0]}\nA: {first_qa[1]}" if first_qa[0] else "",
            )
            .replace(
                "{second_question_and_answer}",
                f"Q: {second_qa[0]}\nA: {second_qa[1]}" if second_qa[0] else "",
            )
        )

        response = model.generate_content(prompt, generation_config=generation_config)
        question = response.text.strip().replace("Q:", "").strip()

        print(f"\n🤖 Q{turn+1}: {question}")
        answer = input("👤 Your answer: ").strip()

        if not answer:
            answer = "No specific preference"
            print(f"   (Using default: {answer})")

        qa_pairs.append((question, answer))

    clarification_block = "\n\n".join([f"Q: {q}\nA: {a}" for q, a in qa_pairs])

    # Generate augmented stated_intentions
    print(f"\n🔄 Generating augmented intentions based on your answers...")

    augment_prompt = augment_prompt_template.replace(
        "{stated_intention}", stated_intention
    ).replace("{clarification_questions_and_answers}", clarification_block)

    try:
        response = model.generate_content(
            augment_prompt, generation_config=generation_config
        )
        response_str = response.text if hasattr(response, "text") else "{}"
        response_str = re.sub(r"```(?:json)?", "", response_str).strip()

        if response_str:
            data_dict = json.loads(response_str)
            sorted_list = [data_dict[str(i)] for i in range(1, 11)]

            print(f"\n✅ Generated {len(sorted_list)} augmented intentions:")
            for i, intention in enumerate(sorted_list, 1):
                print(f"   {i:2d}. {intention}")

            final_output[task_id] = {
                "clarification_QAs": clarification_block,
                "augmented_intentions": sorted_list,
            }
        else:
            print("❌ Empty response from API")
            final_output[task_id] = {
                "clarification_QAs": clarification_block,
                "augmented_intentions": [stated_intention] * 10,
            }
    except Exception as e:
        print(f"❌ Error generating augmented intentions: {e}")
        final_output[task_id] = {
            "clarification_QAs": clarification_block,
            "augmented_intentions": [stated_intention] * 10,
        }

    # Save after each task (in case of interruption)
    with open(output_file, "w") as json_file:
        json.dump(final_output, json_file, indent=4)

    print(
        f"\n💾 Progress saved ({len(final_output)}/{len(stated_intentions)} completed)"
    )
    print("=" * 60)

print(f"\n🎉 All clarifications completed!")
print(f"   Total tasks: {len(stated_intentions)}")
print(f"   Completed clarifications: {len(final_output)}")
print(f"   Output file: {output_file}")
