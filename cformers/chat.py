import argparse
from interface import AutoInference as AI
import sys

model_map = {
    'pythia': 'OpenAssistant/oasst-sft-1-pythia-12b', 
    'bloom': 'bigscience/bloom-7b1', 
    'gptj': 'EleutherAI/gpt-j-6B', 
    'stability': 'stabilityai/stablelm-tuned-alpha-7b'
}

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prompt", nargs="+",help="Enter a value for the prompt")
parser.add_argument("-t", "--tokens",help="Number of tokens to generate",type=int, default=100)
parser.add_argument("-m", "--model", help="Specify a Model", choices=model_map.keys(),default="pythia")
args = parser.parse_args()

def generate(prompt,arg=args):
    # print('Model is '+arg.model)
    if arg.model == 'stability':
        prompt = f"<|SYSTEM|> <|USER|>{prompt}<|ASSISTANT|>"
        def hook(text):
            print(text, end='')
            sys.stdout.flush()
        x = ai.generate(prompt, num_tokens_to_generate=arg.tokens, temperature=0.7, print_streaming_output=False, streaming_token_str_hook=hook)
    elif arg.model == 'pythia':
        x = ai.generate("<|prompter|>"+prompt+"<|endoftext|><|assistant|>", num_tokens_to_generate=arg.tokens)
    elif arg.model == 'bloom':
        x = ai.generate(""+prompt+"", num_tokens_to_generate=arg.tokens)
    elif arg.model == 'gptj':
        x = ai.generate(""+prompt+"", num_tokens_to_generate=arg.tokens)
    else:
        x = ai.generate("<|prompter|>"+prompt+"<|endoftext|><|assistant|>", num_tokens_to_generate=arg.tokens)
    return x

ai = AI(model_map[args.model])

if not args.prompt:
    while True:
        my_prompt = input("> ")
        if my_prompt.lower() == 'exit':
            break
        try:
            x = generate(my_prompt,args)
        except KeyboardInterrupt:
            print('—')
            continue
        # print(x['token_str'])
else:
    my_prompt = ' '.join(args.prompt)
    generate( my_prompt,args)
    # print(x['token_str'])
