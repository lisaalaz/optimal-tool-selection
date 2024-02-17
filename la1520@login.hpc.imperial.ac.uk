import re
from funchub.math import *

def func_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, return_top=5):
    # Inference mode for funcqa and gsm8k-xl

    # immediately set current generation to the empty string
    cur_generation = ""
    # and set current generation with function to the empty string
    cur_generation_with_func = ""
    start_length = []
    end_length = []
    logs = []

    debug_log = []
    debug_log.append(f"{case_idx}\n")

    funcmodel.inference_mode = "func_embedding"

    # list the available functions (tools) for the current task
    func_map = list(funcmodel.func_dict.keys())
    
    # Note that if this try-except fails the current generation will remain the empty string
    try:
        results = [] 
        func_calls = []

        #loop_count = 1
        while True: # loop until break
            
            # The general few-shot template simply asks to answer the question step by step (does not ask for a specific tool). 
            # We insert the current question into it and append the current generation to it (i.e. after "Answer:")
            # (note that at the beginning the current generation is empty)
            prompt = templates["general"].replace("[QUESTION]", question) + cur_generation

            # We take the prompt composed above, pass it into the model and assign the generated sequence to the results variable
            # we pass '\n' as stop token here, and set return_top to 5 (so results is actually a tuple of (decoded, generation_log))
            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top)

            # return_top is a flag that can be set. The default is 5.
            if return_top > 0:
                results, token_log = results # we decouple the results tuple
                logs.append(token_log)
            endflag = True
            current_token = 0
            record_tokens = token_log[-1] # this is the last generated token

            # now take the current generation (from the decoded list assigned to results) and eliminate the prompt part that asked to solve step by step, keeping only the generated answer
            cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")
            
            for op in func_map:
                if cur_generation.endswith(op+"("):
                    endflag = False # set the outer loop not to break if the generation ends with toolken and (
                    
                    if start_length and end_length:
                        bias = 0

                        # copy the current generation to cur_generation_with_func
                        cur_generation_with_func = cur_generation
                        for i in range(len(start_length)):
                            cur_generation_with_func = cur_generation_with_func[:start_length[i]+bias] +func_calls[i] + cur_generation_with_func[end_length[i]+bias:]
                            bias += len(func_calls[i]) - (end_length[i] - start_length[i])
                    
                    else:
                        cur_generation_with_func = cur_generation # take the generated answer, which contains the function
                    prompt = templates[op].replace("[QUESTION]", question) + cur_generation_with_func # now complete the prompt with exemplars for that operation, adding the question and generation so far (up to the tool)
                    len_prompt = len(prompt)
                    funcmodel.inference_mode = "baseline" # this ensures we won't generate any toolkens when adding the arguments below
                    results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[29897, 3892], return_top=return_top) # pass the prompt to add the arguments, here the stop token is ) or )=

                    funcmodel.inference_mode = "func_embedding" # now we can return to reasoning mode to continue the generation (and potentially add more toolkens) later
                    if return_top > 0:
                        results, token_log = results
                        logs.append(token_log)
                    generated = results[0][len_prompt:] # we just take the arguments just generated
                    cur_generation += generated # and extend the generation with them
                    debug_log.append(f"the token_log is: {token_log}\n")
                    debug_log.append(f"the arguments inserted are: {generated}\n")

                    # now let us just isolate the arguments
                    args = cur_generation.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")")
                    # remove any $ in the args
                    args = args.replace("$", "")
           
                    if ", " in args:
                        args = args.replace(", ", ";").replace(",", "").replace(";", ", ") # this leaves ", " unchanged but eliminates commas without spaces
                    args = args.replace(" ", "")

                    if "(" not in args or ")" not in args:
                        raise Exception("invalid args")
                    # handle %
                    if '%' in args:
                        temp = args.split("(")[1].split(")")[0].split(",")
                        for arg_i, arg in enumerate(temp):
                            # if have percentage, convert to decimal
                            if "%" in arg:
                                arg = arg.replace("%", "").strip()
                                arg = str(float(arg) / 100)
                            temp[arg_i] = arg
                        args = f"({', '.join(temp)})"
                                
                    try: 
                        
                        res = eval(f"{op[1:-1]}_{args}") # evaluate function with args
                        func_calls.append(f"{op}{args} = {res}") # append all to func_calls list to output it later
                        start_length.append(len(cur_generation.split(op)[0]))
                        cur_generation = cur_generation.split(op)[0] + str(res)
                        end_length.append(len(cur_generation))

                        # only generate the next token
                        # disable all the numbers
                        # continue generation
                        prompt = templates["general"].replace("[QUESTION]", question) + cur_generation

                        results = funcmodel.generate([prompt], max_gen_len=1, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top, disable_token = [29900, 29896, 29906, 29941, 29946, 29945, 29953, 29955, 29947, 29929]) # disable all the numbers: 0-9
                                                
                        if return_top > 0:
                            results, token_log = results
                            logs.append(token_log)
                        cur_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "")
                    except:
                        # backtrace 
                        current_token += 1
                        decode_token = lambda x: funcmodel.tokenizer.decode(x) if x < 32000 else func_map[x - 32000]
                        cur_generation = cur_generation.split(op)[0] + decode_token(record_tokens[1][current_token][0])
                    break
            #loop_count += 1    
            if endflag: # if a toolken has not been generated at this iteration, we break and conclude the inference of the current example
                break

        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": "success"
        }

    except Exception as e:
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }

    text_file = open("debug_log.txt", "a+")
    for line in debug_log:
        text_file.write(line)
    text_file.close()


    # text_file = open("debug_log.txt", "a+")
    # for line in debug_log:
    #     text_file.write(line)
    # text_file.close()

    return log


def vh_embedding_inference(case_idx, question, funcmodel, temperature, top_p, max_func_call):
    funcmodel.inference_mode = "func_embedding"
    inputs = question[0]
    disable_funcs = question[1]
    last_func = []
    for _ in range(max_func_call):
        inputs = funcmodel.generate([inputs], max_gen_len=1, temperature=temperature, top_p=top_p,return_top=0, disable_func=disable_funcs + last_func, no_left_parens=True)[0]

        if inputs.endswith(">"):
            inputs = inputs.replace("]<", "] <")
            inputs += '\n'
            last_func = [] if "[WALK]" in inputs.split("\n")[-2] else re.findall(r"\[.*?\]", inputs.split("\n")[-2])
            print("last func", last_func)
        if "[END]" in inputs.split("Plan:")[-1]:
            break
    

    log = {
    "case_idx": case_idx,
    "question": question[0],
    "func_calls": inputs.replace(question[0], "").strip().split("\n"),
    "generation": inputs.replace("\n", "\\n").strip(),
    # no need to return logs
    # "token_log": logs,
    "status": "success"
    }
    return log


def kamel_embedding_inference(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, max_func_call):

    funcmodel.inference_mode = "func_embedding"
    cur_generation = ""
    if "funcgeneral" not in templates:
        templates["funcgeneral"] = templates["general"]
    try:
        results = []
        func_calls = []
        while True:
            if max_func_call == 0:
                break
            prompt = templates["funcgeneral"].replace("[QUESTION]", question) + cur_generation

            results = funcmodel.generate([prompt], max_gen_len=max_gen_len, temperature=temperature, top_p=top_p, stop_token=[13])
            max_func_call -= 1
            
            cur_generation = results[0].replace(templates["funcgeneral"].replace("[QUESTION]", question), "")
            # one function token is enough
            break
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": "success"
        }
        # f.write(json.dumps(log) + "\n")

    except Exception as e:
        # if local_rank == 0:
        log = {
            "case_idx": case_idx,
            "question": question,
            "func_calls": func_calls,
            "generation": cur_generation.replace("\n", "\\n").strip(),
            "status": str(e)
        }
    return log