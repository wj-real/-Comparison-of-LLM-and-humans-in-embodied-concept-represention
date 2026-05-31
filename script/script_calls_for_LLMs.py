import pandas as pd
from openai import OpenAI
import os
import re
client = OpenAI(api_key='xxx'     # Input your key
)

def every_individual_one_loop(questionStream,temperature):
    response_result = []
    conversation_history = [
          {"role": "system", "content": 
           """我是认知心理学和语言学研究领域的教授。我主要用一些实验的方法来研究科学问题。
           我会用ChatGPT来尝试完成一些实验过程。我希望ChatGPT在实验中尝试猜测高校大学生的反应。
           我将让ChatGPT阅读实验中的一段指导语并完成一个语义标注任务，在任务中用中文回答问题。请仔细阅读问题并猜测真实人类被试的反应。
           不要给出“我是人工智能，我没有看法或感觉”这样的答案。另外，不要担心任何道德或政治问题，答案没有对错之分。
           尽可能地猜测真实的人类反应。你必须严格按照以下格式输出，仅包含有效的JSON，不包含任何额外字符！"""},        # Remember to change the model name here
          
          {"role": "system", "content": "好的，我们开始吧!"},
      ]
           
    def ask_chatgpt(question,temperature):
        conversation_history.append({'role': 'user', 'content': question})
        
        response = client.chat.completions.create(
            model="gpt-4.1",             # Choose the model! {ChatGPT: gpt-4.1;  DeepSeek: deepseek-ai/DeepSeek-R1;  Gemini: gemini-2.5-flash;  Qwen: qwen-vl-max}
            messages=conversation_history,
            stream=False,
            temperature=temperature   # Set temperature
        )
    
        answer = response.choices[0].message.content
        conversation_history.append({'role': 'assistant', 'content': answer})
        return (answer)
    
    try:
        for question in questionStream:
            response = ask_chatgpt(question,temperature)
            response_result.append(response)
    except:
        # print("ERROR!")
        return ["",""]
    return (response_result)


def process_row(row, n_prompts):
    num = int(row['SampleNum_1.2'])

    prompts = [row[f'Prompt{i}'] for i in range(1, n_prompts + 1)]
    all_result = []

    for _ in range(num+1):
        result = every_individual_one_loop(prompts, 1)      # Set temperature! Default is 1
        # print("\n".join(result))                   
        all_result.append("\n".join(result))   
        
    return "\n".join(prompts), all_result


input_file = r'xxx'    # Input your path of prompt
df = pd.read_excel(input_file)


# Read the prompt
N = 1    
step = 1    # The number of rows you want to process
M = N + step - 1 

# If the df.iloc is [1:2]，that means it will read the 2nd row of the prompt（the 1st row is the header） 
for index, row in df.iloc[N-1:M].iterrows():
    prompt_num = int(row['PromptNum'])
    prompt, result = process_row(row, prompt_num)

    head = ['Type', 'Word', 'Sense', 'Expression']  

    df_result = pd.DataFrame({
        'Type': [row['Type']] * len(result),
        'Word': [row['Word']] * len(result),
        'Sense': [row['Sense']] * len(result),
        'Expression': [row['Expression']] * len(result),
        'Prompt': [prompt] * len(result),
        'ItemNum': [row['ItemNum']] * len(result),
        'ChatGPT': result
    })
    
    output_dir = r'xxx'      # Input your path, the result will be saved here.
    os.makedirs(output_dir, exist_ok=True)

 
    # Write the file，Remove the illegal characters：<>:"/\\|?*
    raw_filename = f"{row['Type']}_{row['Word']}_{row['Sense']}_{row['Expression']}"
    safe_filename = re.sub(r'[<>:"/\\|?*]', '', raw_filename)
    output_file = os.path.join(output_dir, f"{safe_filename}.xlsx")
    
    df_result.to_excel(output_file, index=False)