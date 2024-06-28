import string
import openai
import time

def get_checked_prediction(cur_prompt,check_args,check_func,model,max_try=1):
    msg = "ERROR"
    pred = None
    this_prompt = cur_prompt
    try_count = 0
    while msg != "" and try_count < max_try:
        response = get_inference_response(this_prompt,model=model)
        try_count += 1
        if response is not None:
            pred = response["choices"][0]["message"]["content"]
            msg = check_func(check_args,pred)
            query_msg = "\nThe above response may have errors. Please provide another response on the following feedback:\n"
            #this_prompt += "\n" + pred + query_msg + msg
            this_prompt = cur_prompt + "\n" + pred + query_msg + msg
            #if msg != "":
            #    print(msg,pred)
        else:
            #reset the prompt due to error from exceeding max context length
            this_prompt = cur_prompt
    return pred, msg == ""

def get_inference_response(prompt,model):
    #model options: gpt-3.5-turbo-0125, gpt-4-0125-preview
    print(prompt)
    retry = True
    while retry:
        try:
            response = openai.ChatCompletion.create(
                model=model,
                messages=[
                    {"role": "user", "content": prompt},
                ],
                response_format={ "type": "json_object" },
                request_timeout=5*60,
                temperature=0.75,
            )
            """
            response = openai.Completion.create(
                model="gpt-3.5-turbo-instruct",
                prompt=prompt
            )
            """
            retry = False
        except openai.error.APIError as e:
            print(f"OpenAI API returned an API Error: {e}")
            time.sleep(2)
        except openai.error.ServiceUnavailableError as e:
            print(f"OpenAI API returned a Service Unavailable Error: {e}")
            time.sleep(2)
        except Exception as e:
            print(f"OpenAI API returned an Exception: {e}")
            if "maximum context length" in str(e):
                print("breaking out!")
                response = None
                break
            time.sleep(2)
    return response

def is_str_equal(str1,str2):
    translator = str.maketrans('', '', string.punctuation)
    return str1.lower().translate(translator).split() == str2.lower().translate(translator).split()

def is_str_in_str(str1,str2):
    translator = str.maketrans('', '', string.punctuation)
    return set(str1.lower().translate(translator).split()) <= set(str2.lower().translate(translator).split())
    