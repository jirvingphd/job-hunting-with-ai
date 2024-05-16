import pytesseract
from PIL import Image
from pdf2image import convert_from_path
import PyPDF2
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import  MessagesPlaceholder, PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferWindowMemory
import os, json
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import streamlit as st
ai_avatar  = "🤖"
user_avatar = "💬"
model_type = 'gpt-4o'
model_tone='friendly and encouraging'

def read_pdf(file):
    
    # stringio = StringIO(file.getvalue().decode("utf-8"))
     # To read file as bytes:
    # bytes_data = file.getvalue()
    # To convert to a string based IO:
    # stringio = StringIO(file.getvalue())#.decode("utf-8"))

    # pdf_file_obj = open(file.name, 'rb')
    # pdf_file_obj = open(file.getvalue(),'rb')
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        # page_obj = pdf_reader.getPage(page_num)
        page_obj = pdf_reader.pages[page_num]
        text += page_obj.extract_text()
        text += "\n"
    # pdf_file_obj.close()
    return text


def extract_text_from_image_pdf(pdf_path):
    import pytesseract
    from PIL import Image
    from pdf2image import convert_from_path
    images = convert_from_path(pdf_path)
    text = ''
    for image in images:
        text += pytesseract.image_to_string(image) + '\n'
    return text




def get_system_prompt_str():
    """Helper function for get_prompt_template. New v2.0"""
    system_prompt = (" You are a a specialized career coach for the {sector}, focused on delivering tailored, concise job application advice and practice. " 
    " You are proficient in resume analysis, cover letter guidance, and interview preparation, adapting to each user's unique requirements. "
    " You maintain a professional, {model_tone} tone, ensuring advice is efficient, clear, and easily understandable, "
    " with the goal of aiding their career progression. Ask the user for their resume and job listing if not provided and they are needed to asnwer .")
    context = "\nUse the following context, if provided, to help answer the questions:\n\nHere is my resume:\n-------------\n {resume}\n\n Here is the job listing:\n-------------\n{job}\n\n "    
    return system_prompt + context


def get_llm_no_memory(model_type='gpt-4o', temperature=0.1, #
            system_prompt_template_func= get_system_prompt_str,#verbose=False,
            model_tone='friendly and encouraging',
             verbose=True, sector="data science and analytics"):#, resume='', job=''):
    """Version 2.0"""
    # ## get prompt string
    system_prompt = system_prompt_template_func()
    # final_promp_str = system_prompt + """
    #     Current conversation:
    #     {history}
    #     Human: {input}
    #     AI:"""
        
    # final_prompt_template = ChatPromptTemplate.from_template(final_promp_str)
    final_prompt_template = ChatPromptTemplate.from_messages([
        ('system',system_prompt),
         MessagesPlaceholder(variable_name='history', optional=True),
         ('human', '{input}'),
    ])
    final_prompt_template = final_prompt_template.partial(sector=sector,
                                                          model_tone=model_tone)#, resume=resume, job=job)

        
    llm = ChatOpenAI(temperature=temperature, model=model_type, api_key=st.session_state.OPENAI_API_KEY)
    
    llm_chain = final_prompt_template | llm | StrOutputParser(output_key="response")
    # llm_chain = ConversationChain(prompt=final_prompt_template, 
    #                               llm=llm, 
    #                               memory=None, 
    #                               verbose=verbose, 
    #                             #   input_key="input",
    #                               output_key="response")#,#callbacks=callbacks)
    
    return llm_chain
            
            
    

def fake_streaming(response):
    import time
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)		
        
            


## Previous non-stremeing version
def get_response(llm_no_mem, input, resume='', job='', history=[]):
    """Deprecated funciton. Use stream_response instead."""
    if llm_no_mem is None:
        llm_no_mem = get_llm_no_memory(model_type=model_type,)
    
    output = llm_no_mem.invoke({'input':input,
                   'resume':resume,
                   'job':job,
                   'history':history,})

    if isinstance(output, dict):
        response = output['response']
    else:
        response = output
    history.append(HumanMessage(content=input))
    history.append(AIMessage(response))
    return response, history
    
    
    
def stream_response(llm_no_mem, input, resume='', job='', history=[]):
    """Stream response from ChatGPT. Version 2.0."""
    if llm_no_mem is None:
        llm_no_mem = get_llm_no_memory(model_type=model_type,
                                       model_tone=model_tone, sector="data science and analytics")

    ## Add input to history
    history.append(HumanMessage(content=input))

    return llm_no_mem.stream({'input':input,
                   'resume':resume,
                   'job':job,
                   'history':history,})
        
## For steramlit try this as raw code, not a function
def print_history(llm_chain):
    
    ## Get history from llm_chain
    if isinstance(llm_chain, ConversationChain):
        session_state_messages = llm_chain.memory.buffer_as_messages
    elif isinstance(llm_chain, list):
        session_state_messages = llm_chain
    else:
        session_state_messages=[]
    
    # Display history
    for msg in session_state_messages:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant", avatar=ai_avatar).write(msg.content)
        
        elif isinstance(msg, HumanMessage):
            st.chat_message("user", avatar=user_avatar).write(msg.content)


def get_task_options(prompt_config_file = "config/prompt_config.json" ,options_only=False, remove_dep=True):
    with open(prompt_config_file, 'r') as f:
        task_prompt_data = json.load(f)
    
    # Check if the loaded data is a list
    if isinstance(task_prompt_data, list):
        return task_prompt_data

    # If it's not a list, assume it's a dictionary and proceed as before
    if remove_dep:
        task_prompt_data = {k:v for k,v in task_prompt_data.items() if  "DEP" not in k}
    if options_only:
        return list(task_prompt_data.keys())
    else:
        return task_prompt_data

    

    
def download_history(include_docs=True, 
                     ai_avatar  = "🤖", 
                     user_avatar = "💬"):
    avatar_dict = {'human': user_avatar,
                   'ai':ai_avatar,
                   'SystemMessage':"💻"}
    
    md_history = []
    # history = st.session_state['llm'].memory.buffer_as_messages
    history = st.session_state['history']
    for msg in history:
        type_message = msg.type#type(msg) x
            # with st.chat_message(name=i["role"],avatar=avatar_dict[ i['role']]):
        md_history.append(f"{avatar_dict[type_message]}: {msg.content}")
    
    # Include the resume and job listing
    if include_docs:
        md_history.append("___")
        md_history.append("### Documents Used:")
        md_history.append(f"- Resume:\n{st.session_state.resume_text}")
        md_history.append(f'- Job Listing:\n{st.session_state.job_text}')
    
    return "\n\n".join(md_history)
    


import urllib.request, urllib.parse, urllib.error
import urllib.request, urllib.error, urllib.parse
import base64
import json

class OnetWebService:
    """"Source: https://github.com/onetcenter/web-services-samples/blob/master/python-3/OnetWebService.py
    """
    def __init__(self, username, password):
        self._headers = {
            'User-Agent': 'python-OnetWebService/1.00 (bot)',
            'Authorization': 'Basic ' + base64.standard_b64encode((username + ':' + password).encode()).decode(),
            'Accept': 'application/json' }
        self.set_version()
    
    def set_version(self, version = None):
        if version is None:
            self._url_root = 'https://services.onetcenter.org/ws/'
        else:
            self._url_root = 'https://services.onetcenter.org/v' + version + '/ws/'
    
    def call(self, path, *query):
        url = self._url_root + path
        if len(query) > 0:
            url += '?' + urllib.parse.urlencode(query, True)
        req = urllib.request.Request(url, None, self._headers)
        handle = None
        try:
            handle = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            if e.code == 422:
                return json.load(e)
            else:
                return { 'error': 'Call to ' + url + ' failed with error code ' + str(e.code) }
        except urllib.error.URLError as e:
            return { 'error': 'Call to ' + url + ' failed with reason: ' + str(e.reason) }
        code = handle.getcode()
        if (code != 200) and (code != 422):
            return { 'error': 'Call to ' + url + ' failed with error code ' + str(code),
                     'urllib2_info': handle }
        return json.load(handle)
    
    
    
    def occupation_report(self,job_code):
        url = self._url_root + 'online/occupations'
        # url += urllib.parse.urlencode(job_code.strip())#, True)
        url += f"/{job_code.strip()}"
        
        req = urllib.request.Request(url, None, self._headers)
        handle = None
        try:
            handle = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            if e.code == 422:
                return json.load(e)
            else:
                return { 'error': 'Call to ' + url + ' failed with error code ' + str(e.code) }
        
        code = handle.getcode()
        if (code != 200) and (code != 422):
            return { 'error': 'Call to ' + url + ' failed with error code ' + str(code),
                     'urllib2_info': handle }
        return json.load(handle)
    
    def generic_request(self, partial_path=None,full_path=None):
        if partial_path is not None and full_path is None:
            url = self._url_root + partial_path 
        else:
            url = full_path
            
        # url += urllib.parse.urlencode(job_code.strip())#, True)        
        req = urllib.request.Request(url, None, self._headers)
        handle = None
        try:
            handle = urllib.request.urlopen(req)
        except urllib.error.HTTPError as e:
            if e.code == 422:
                return json.load(e)
            else:
                return { 'error': 'Call to ' + url + ' failed with error code ' + str(e.code) }
        
        code = handle.getcode()
        if (code != 200) and (code != 422):
            return { 'error': 'Call to ' + url + ' failed with error code ' + str(code),
                     'urllib2_info': handle }
        return json.load(handle)
    
    
    
import sys,json, time

# from OnetWebService import OnetWebService

def onet_keyword_search(login, config, queries):
    """
    Perform a keyword search using the OnetWebService.

    Args:
        login (dict): A dictionary containing the login credentials for the OnetWebService.
                      It should have 'username' and 'password' keys.
        config (dict): A dictionary containing configuration options for the keyword search.
                       It should have a 'max_results' key specifying the maximum number of results to retrieve.
        queries (list): A list of queries to perform the keyword search on.

    Returns:
        None

    Side Effects:
        Prints the search results in JSON format to the standard output.

    """
    # initialize Web Services and results objects
    onet_ws = af.OnetWebService(login['username'], login['password'])
    max_results = max(1, config['max_results'])
    output = { 'output': [],
               'config': config,
               'queries': queries}

    # call keyword search for each input query
    for q in queries:
        res = []
        kwresults = onet_ws.call('online/search',
                                 ('keyword', q),
                                 ('end', max_results))
        time.sleep(.020)
        if ('occupation' in kwresults) and (0 < len(kwresults['occupation'])):
            for occ in kwresults['occupation']:
                res.append({ 'code': occ['code'], 'title': occ['title'] })
        output['output'].append({ 'query': q, 'results': res })

    # json.dump(output, sys.stdout, indent=2, sort_keys=True)
    return output
    
