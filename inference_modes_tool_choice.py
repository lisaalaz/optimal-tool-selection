import re
import random
import numpy as np
from funchub.math import *
def func_embedding_inference_tool_choice(templates, case_idx, question, funcmodel, doc_dict, exemplar_dict, temperature, top_p, max_gen_len, return_top=10, 
                                         hints_pos="start", decode_all=False, docs=False):
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
    #funcmodel.inference_mode = "func_embedding"
    # list the available functions (tools) for the current task
    func_map = list(funcmodel.func_dict.keys())
    decode_all_string = "_decodeall" if decode_all else ""
    docs_string = "_docs" if docs else ""
    endflag = False
    
    # Note that if this try-except fails the current generation will remain the empty string
    try:
        results = [] 
        func_calls = []
        loop_count = 1
        while True: # loop until break
            funcmodel.inference_mode = "func_embedding"
            debug_log.append(f"We are at generation loop n.{loop_count}!\n")
            # The general few-shot template simply asks to answer the question step by step (does not ask for a specific tool). 
            # We insert the current question into it and append the current generation to it (i.e. after "Answer:")
            # (note that at the beginning the current generation is empty)
            prompt = templates["general"].replace("[QUESTION]", question) + cur_generation
            debug_log.append(f"The question is {question}\n")
            #debug_log.append(f"We pass a general prompt into the model to make it reason\n")
            #debug_log.append(f"{prompt}\n")
            # We take the prompt composed above, pass it into the model and assign the generated sequence to the results variable
            # we pass '\n' as stop token here, and set return_top to 5 (so results is actually a tuple of (decoded, generation_log))
            results = funcmodel.generate([prompt], max_gen_len=128, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top)
            # return_top is a flag that can be set. The default is 5.
            if return_top > 0:
                results, token_log = results # we decouple the results tuple
                logs.append(token_log)

            new_generation = results[0].replace(templates["general"].replace("[QUESTION]", question), "").replace(cur_generation, "")           
            #debug_log.append(f"And the new_generation is: {new_generation}\n")

            #endflag = True
            current_token = 0
            record_tokens = token_log[-1] # this is the last generated token
            # now take the current generation (from the decoded list assigned to results) and eliminate the prompt part that asked to solve step by step, keeping only the generated answer
            
            found_toolkens = any([x[0] >= 32000 for x in token_log])
            debug_log.append(f"found_toolkens is {found_toolkens}!\n")
            if found_toolkens and "####" not in new_generation: # check that a tool has been generated and that we have not reached the end result
              
                cur_generations = []
                if decode_all:
                  for i in range(len(token_log)):
                    for t in token_log[i][1]:
                      if (t[0] >= 32000):
                          toolken = t[0]
                          #debug_log.append(f"the toolken is {toolken}\n")
    
                          token_list = [x[0] for x in token_log[:i]] + [toolken]
                          #debug_log.append(f"the token_list is {token_list}\n")
                          #debug_log.append(f"we have a toolken!\n")
                          generation = funcmodel.decode_list(token_list)
                          #debug_log.append(f"generation from token list is: {generation}\n")

                          #if loop_count > 1:

                          # suppose we have "so she has found <add>("
                          # but also found a toolken earlier, so we also have "so she has <multiply>("

                          generation = cur_generation + generation
                          #debug_log.append(f"cur_generation now is: {generation}\n")

                          #pos = i
                          cur_generations.append(generation)
                else:
                    toolkens = []
                    i = np.where(np.array([x[0] for x in token_log]) >= 32000)[0][0]
                    for t in token_log[i][1]:
                        if t[0] >= 32000:
                            toolkens.append(t[0])
                    #debug_log.append(f"the toolkens we found are {toolkens}\n\n")
                    record_tokens = token_log[-1] # this is the last generated token with all its top k options
                    token_list = [x[0] for x in token_log[:-1]]
                    for t in toolkens:
                        list_to_decode = token_list + [t]
                        gen = funcmodel.decode_list(list_to_decode)
                        debug_log.append(f"gen is: {gen}\n")  
                        
                        if loop_count > 1:
                            gen = cur_generation.split("<")[0] + gen
                        #debug_log.append(f"gen with previous part is: {gen}\n")    
                        
                        cur_generations.append(gen)
                #if not cur_generations:
                #    cur_generations = [cur_generation]
                debug_log.append(f"cur_generations is: {cur_generations}\n")                
              
                all_generations = []
                operations = []
                #debug_log.append(f"Now we append each potential toolken to what we generated before. And look at each of these options to insert the arguments.\n\n")  
                for cur_generation in cur_generations:
                #debug_log.append(f"Looking at this option:\n{cur_generation}\n")  
              
            
                  for op in func_map:
                    if cur_generation.endswith(op+"("):
                      #debug_log.append(f"the operation found is: {op}\n")
                      #endflag = False # set the outer loop not to break if the generation ends with toolken and (
                    
                      if start_length and end_length:
                         #  debug_log.append(f"we have start length {start_length} and end length {end_length}\n")
                         bias = 0
                         # copy the current generation to cur_generation_with_func
                         cur_generation_with_func = cur_generation
                         #  debug_log.append(f"the cur_generation_with_func is {cur_generation_with_func}\n")
                   
                      else:
                         #  debug_log.append(f"we dont have start length and end length\n")
                         cur_generation_with_func = cur_generation # take the generated answer, which contains the function
                         #  debug_log.append(f"the cur_generation_with_func (with 'else' statement) is {cur_generation_with_func}\n")
                    
                      funcmodel.inference_mode = "baseline" # this ensures we won't generate any toolkens when adding the arguments below
                      prompt = templates[op].replace("[QUESTION]", question) + " " + cur_generation_with_func # now complete the prompt with exemplars for that operation, adding the question and generation so far (up to the tool)
                      #debug_log.append(f"Now we pass this prompt into the model to generate the arguments:\n{prompt}\n")   
                      len_prompt = len(prompt)
                      results = funcmodel.generate([prompt], max_gen_len=32, temperature=0, top_p=top_p, stop_token=[29897, 3892], return_top=return_top) # pass the prompt to add the arguments, here the stop token is ) or )= 
                      if return_top > 0:
                         results, token_log = results
                         logs.append(token_log)     
                      generated = results[0][len_prompt:] # we just take the arguments just generated
                      #debug_log.append(f"the generated arguments are: {generated}\n")
                      cur_generation += generated # and extend the generation with them
                      # if (not cur_generation.endswith(")=") and not cur_generation.endswith(")")):
                      #     cur_generation += ")"
                      # debug_log.append(f"the cur_generation after argument insertion is:\n{cur_generation}\n")
                      # now let us just isolate the arguments
                      args = cur_generation.split(op)[-1].replace("=", "").replace(">", "").replace("((", "(").replace("))", ")")
                      # remove any $ in the args
                      args = args.replace("$", "")
           
                      if ", " in args:
                        # we also handle ^ here
                        args = args.replace(", ", ";").replace(",", "").replace(";", ", ").replace("^", "**") # this leaves ", " unchanged but eliminates commas without spaces
                      args = args.replace(" ", "")
                      if "(" not in args or ")" not in args:
                        raise Exception("invalid args")
                      # handle %
                      if '%' in args:
                        temp = args.split("(")[1].split(")")[0].split(",")
                        #debug_log.append(f"temp is {temp}")
                        for arg_i, arg in enumerate(temp):
                            #debug_log.append(f"arg is {arg}")
                            # if have percentage, convert to decimal
                            if "%" in arg:
                                arg = arg.replace("%", "").strip()
                                arg = str(float(arg) / 100)
                                #debug_log.append(f"arg now is {arg}")
                            temp[arg_i] = arg
                        args = f"({', '.join(temp)})"
                        #debug_log.append(f"args is {args}")
                    
                      if (
                         op not in ["<log>", "<ln>", "<sqrt>"] and "," not in args
                        ) or (
                            op in ["<choose>", "<permutate>", "<remainder>", "<lcm>", "<gcd>"] and "." in args
                            ):
                        debug_log.append(f"operation {op}{args} is not well formed!\n")
                      else:
                        try: 
                          res = eval(f"{op[1:-1]}_{args}") # evaluate function with args
                          #debug_log.append(f"The result is:\n{res}\n")

                          if len(res) < 20: # eliminate ops with too long results as they fill the context window

                            func_calls.append(f"{op}{args} = {res}") # append all to func_calls list to output it later
                            start_length.append(len(cur_generation.split(op)[0]))
                            cur_generation += str(res) # changed this to keep the operation and arguments in the prompt (for the model to choose)
                            #debug_log.append(f"After evaluating the arguments, we obtain:\n{cur_generation}\n")
                            end_length.append(len(cur_generation))
                            all_generations.append(cur_generation)
                            operations.append(op)
                        except:
                          # backtrace 
                          current_token += 1
                          decode_token = lambda x: funcmodel.tokenizer.decode(x) if x < 32000 else func_map[x - 32000]
                          cur_generation = cur_generation.split(op)[0] + decode_token(record_tokens[1][current_token][0])
                          #debug_log.append(f"Exception! could not evaluate the arguments. So we generated this and we won't use it as a hint.\n")
                    
                debug_log.append(f"len(all_generations) is {len(all_generations)}\n")
                if len(all_generations) == 0:
                   cur_generation = " ".join(all_generations[-1].split("<")[0].split(" ")[:]) #[:-3]
                   #funcmodel.inference_mode = "func_embedding" # return to func embedding mode before next loop
              
                else:  
                   #debug_log.append(f"all_generations is {all_generations}\n")
                   hints = str([*dict.fromkeys(["<" + x.split("<")[-1] for x in all_generations])]).replace("'", "")
                   #debug_log.append(f"the hints are: {hints}\n")
            
                   # always cut off the last 2 words (plus the plaintext operation if present)
                   
                   if " =" in all_generations[0].split("<")[0]:
                      gen_without_hints = " ".join(" ".join(all_generations[-1].split("<")[0].split("=")[:-1]).split(" ")[:-1]) #[:-3]
                   elif "=" in all_generations[0].split("<")[0]:
                      gen_without_hints = " ".join(" ".join(all_generations[-1].split("<")[0].split("=")[:-1]).split(" ")[:]) #[:-3]
                   else:
                      gen_without_hints = " ".join(all_generations[-1].split("<")[0].split(" ")[:-1]) #[:-3] # because of the extra whitespace before <op> when no plaintext operation is present 

                if hints_pos=="start":
                
                  generation_with_options = hints + " " + gen_without_hints
                  debug_log.append(f"the generation_with_options is:\n{generation_with_options}\n")
                
                  if docs:

                    exemplar_type = "decodeall" if decode_all else "start"
                    exemplars = "\n\n".join([exemplar_dict[op][exemplar_type] for op in operations])
                    
                    instructions = "\n".join([doc_dict[op]["overview"] for op in operations])
                    prompt = templates["choicecompletedocs"].replace("[INSTRUCTIONS]", instructions).replace("[EXEMPLARS]", exemplars).replace("[QUESTION]", question).replace("[ANSWER]", generation_with_options)
                    #prompt = templates["choicedocs"].replace("[EXEMPLARS]", exemplars).replace("[QUESTION]", question).replace("[ANSWER]", generation_with_options)
               
                  else:
                    prompt = templates["choicebefore"].replace("[QUESTION]", question).replace("[ANSWER]", generation_with_options)         
            
                else:

                  generation_with_options = gen_without_hints + " " + hints
                  debug_log.append(f"the generation_with_options is:\n{generation_with_options}\n")

                  if docs:
                    #generation_with_options = all_generations[0].split("<")[0] + " " + hints + " " # if end we do not try to elminate the plaintext operation
                    #debug_log.append(f"the generation_with_options is:\n{generation_with_options}\n")
                    exemplars = "\n\n".join([exemplar_dict[op]["end"] for op in operations])
                    instructions = "\n".join([doc_dict[op]["overview"] for op in operations])
                    prompt = templates["choicecompletedocs"].replace("[INSTRUCTIONS]", instructions).replace("[EXEMPLARS]", exemplars).replace("[QUESTION]", question).replace("[ANSWER]", generation_with_options)
                    #prompt = templates["choicedocs"].replace("[EXEMPLARS]", exemplars).replace("[QUESTION]", question).replace("[ANSWER]", generation_with_options)
                  else:
                    #generation_with_options = all_generations[0].split("<")[0] + hints + " "
                    #debug_log.append(f"the generation_with_options is:\n{generation_with_options}\n")
                    prompt = templates["choice"].replace("[QUESTION]", question).replace("[ANSWER]", generation_with_options)
                debug_log.append(f"Now we pass a prompt to help the model choose from the hints.\n")
                debug_log.append(f"{prompt}\n")
                results = funcmodel.generate([prompt], max_gen_len=32, temperature=temperature, top_p=top_p, stop_token=[13], return_top=return_top) # stop at \n
                       
                if return_top > 0:
                  results, token_log = results # we decouple the results tuple
                  logs.append(token_log)
                  #debug_log.append(f"the results before splitting are: {results}\n")
                cur_generation = results[0].split("A: ")[-1].replace(hints + " ", "") #.replace("[", "").replace("<", "")
                debug_log.append(f"the cur_generation after choosing and splitting is:\n{cur_generation}\n")
                #funcmodel.inference_mode = "func_embedding" # return to func embedding mode before next loop
                ######## end added code ########     

            else:
                if "####" in new_generation:
                  endflag=True
                  cur_generation += new_generation
                else:
                   debug_log.append(f"we are in else mode\n")
                   debug_log.append(f"the cur_generation is {cur_generation}\n")
                   debug_log.append(f"the new_generation is {new_generation}\n")
                   debug_log.append("and we put them together and obtain:\n")
                   cur_generation += new_generation
                   debug_log.append(f"{cur_generation}\n")
                 
            loop_count += 1    
            if endflag or loop_count > 10: # if a toolken has not been generated at this iteration, we break and conclude the inference of the current example
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
    text_file = open(f"debug_log_tool_choice_{hints_pos}{docs_string}{decode_all_string}.txt", "a+")
    for line in debug_log:
         text_file.write(line)
    text_file.close()
    return log


def vh_embedding_inference_tool_choice(case_idx, question, funcmodel, temperature, top_p, max_func_call):
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


def kamel_embedding_inference_tool_choice(templates, case_idx, question, funcmodel, temperature, top_p, max_gen_len, max_func_call):

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
