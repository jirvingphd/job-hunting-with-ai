"""APP v2 - functionized resume and job listing """
import json
import streamlit as st
# from langchain import LangChain
import PyPDF2
# import pypdf2 as PyPDF2
from io import StringIO
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import  MessagesPlaceholder, PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationChain
# from langchain.memory import ConversationBufferWindowMemory
import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
import app_functions as af

st.set_page_config(page_title="AI Job Application Assistant", page_icon="💼")#, layout="wide")

## Initialize session state
if 'resume_file' not in st.session_state: 
    st.session_state.resume_file = None

if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
    
if 'job_listing_file' not in st.session_state:    
    st.session_state.job_listing_file = None


if 'job_text' not in st.session_state:
    st.session_state.job_text = None

if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""
    
if "history" not in st.session_state:
    st.session_state['history'] = []


## Display author info
with open("app-assets/author-info.md") as f:
    author_info = f.read()
# st.sidebar.divider()
with st.sidebar.container(border=True):
    # st.subheader("Author Information")
    st.markdown(author_info, unsafe_allow_html=True)


## TITLE
c1, c2 = st.columns([.3,.4])
with c1:
    st.container(height=15, border=False)
    st.image("images/DALLE-2.png", use_column_width=True)
with c2:
    st.title('AI Job Application Assistant')

    st.markdown("*This app uses ChatGPT to analyze your resume and a job listing to provide tailored advice and recommendations, including cover letter creation.*")

# ### HERE
# if 'OPENAI_API_KEY' not in st.session_state:
#     st.session_state['OPENAI_API_KEY'] = ""


        
md_instructions = """

- 👈 **First, open sidebar (`>`) to add your 🔑 OpenAI API Key** and select which ChatGPT model (Default is gpt-4o).
- 👇**Next, open the menu below to 📄Upload Resume and Job Listing**
    - Make sure to press the  `Update Resume` or `Update Job Listing` buttons after uploading or pasting the text.
- 🤖**Finally, select a pre-defined task or ask ChatGPT your own questions.**

- (Optional) 📥*Download the chat history at the end of the session.*

___ 

>- 📺 Video Walkthrough: https://youtu.be/CZrarP5_6Rw


"""
with st.expander("Instructions", expanded=True):
    st.markdown(md_instructions)


## Documents Container
docs_container = st.expander("📄Upload Resume and Job Listing", expanded=False)#border=True)
with docs_container:
    c1, c2 = st.columns([.5,.5])
    c1.markdown("#### Resume")
    resume_container   =    c1.container(border=True)#st.sidebar.container(border=True)
    
    c2.markdown("#### Job Listing")
    # st.sidebar.header("Job Listing")
    job_listing_container = c2.container(border=True)#st.sidebar.container(border=True)

# Upload pdf or paste resume
with resume_container:
    resume_form =st.form(key='resume_form', border=False)
    
    with resume_form:
        st.session_state.resume_file = st.file_uploader("Upload your PDF resume", type="pdf", accept_multiple_files=False)
    
        # Pasted Version
        st.session_state.pasted_resume = st.text_area("or paste your resume here:")
        submit_resume = st.form_submit_button("Update Resume")
    # st.session_state.pasted_resume = st.text_area("or paste your resume here:", height=100,)

# Upload pdf or past job listing    
with job_listing_container:
    job_form =st.form(key='job_form', border=False)

    with job_form:
        
        st.session_state.job_listing_file = st.file_uploader("Upload the PDF job listing", type="pdf", accept_multiple_files=False)
        # Pasted version
        st.session_state.pasted_job_listing = st.text_area("Paste the job listing here", height=100)
        submit_job = st.form_submit_button("Update Job Listing.")
        

## Set resume text
def get_resume_text():
    if st.session_state.pasted_resume != '':
        st.session_state.resume_text = st.session_state.pasted_resume
    elif st.session_state.resume_file is not None:
        st.session_state.resume_text = af.read_pdf(st.session_state.resume_file) 
    else:
        st.error("Please upload a resume or paste it in the text area.")
        
if submit_resume:
    get_resume_text()


## set job listing text
def get_job_text():
    if st.session_state.pasted_job_listing != '':
        st.session_state.job_text = st.session_state.pasted_job_listing
    elif st.session_state.job_listing_file is not None:
        st.session_state.job_text = af.read_pdf(st.session_state.job_listing_file)
    else:
        st.error("Please upload a job listing or paste it in the text area.")
        
if submit_job:
    get_job_text()
    
    

def get_system_prompt_str(with_context=True):
    """Helper function for get_prompt_template. New v2.0"""
    system_prompt = (" You are a a specialized career coach for the {sector}, focused on delivering tailored, concise job application advice and practice. " 
    " You are proficient in resume analysis, cover letter guidance, and interview preparation, adapting to each user's unique requirements. "
    " You maintain a professional, {model_tone} tone, ensuring advice is efficient, clear, and easily understandable, "
    " with the goal of aiding their career progression. Ask the user for their resume and job listing if not provided and they are needed to asnwer .")
    context = "\nUse the following context, if provided, to help answer the questions:\n\nHere is my resume:\n-------------\n {resume}\n\n Here is the job listing:\n-------------\n{job}\n\n "    
    
    if with_context:
        return system_prompt + context
    else:
        return system_prompt


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
            
        
# def fake_streaming(response):
#     import time
#     for word in response.split(" "):
#         yield word + " "
#         time.sleep(.05)		
        
            


# ## Previous non-stremeing version
# def get_response(llm_no_mem, input, resume='', job='', history=[],
#                  model_tone='friendly and encouraging',
#                  model_type='gpt-4o',):
#     """Deprecated funciton. Use stream_response instead."""
#     if llm_no_mem is None:
#         llm_no_mem = get_llm_no_memory(model_type=model_type,
#                                        model_tone=model_tone,
#                                        sector="data science and analytics")
    
#     history.append(HumanMessage(content=input))
    
#     output = llm_no_mem.invoke({'input':input,
#                    'resume':resume,
#                    'job':job,
#                    'history':history,})

#     if isinstance(output, dict):
#         response = output['response']
#     else:
#         response = output
    
#     # history.append(AIMessage(response))
#     return response#, history
    

def stream_response(llm_no_mem, input, resume='', job='', 
                    model_tone='friendly and encouraging',
                    model_type='gpt-4o',
                    history=[]):
    """Stream response from ChatGPT. Version 2.0."""
    if llm_no_mem is None:
        llm_no_mem = get_llm_no_memory(model_type=model_type,
                                       model_tone=model_tone,
                                       sector="data science and analytics")

    ## Add input to history
    history.append(HumanMessage(content=input))

    return llm_no_mem.stream({'input':input,
                   'resume':resume,
                   'job':job,
                   'history':history,})


    
# Settong task and tone    
task_options  = af.get_task_options(prompt_config_file='config/prompt_config.json', options_only=False)
tone_options  = af.get_task_options(prompt_config_file="config/tone_config.json", options_only=False, remove_dep=False)

## OpenAI Model Setting    
with st.sidebar.container(border=True):
    st.subheader("🔑OpenAI API Key")
    st.write('> *Enter your OpenAI API key below and press the `Submit API Key` button. You can sign up for one [here](https://platform.openai.com/api-keys).*')
    
    apikey_form = st.form(key='apikey_form')
    with apikey_form:
        st.session_state['OPENAI_API_KEY']  = st.text_input("OpenAI API Key", type="password", value=st.session_state.OPENAI_API_KEY)
        # with st.sidebar.expander("GPT Model"):
        # st.write('>*Select a GPT model to use for the analysis.*')

        model_tone = st.selectbox("*Select a tone for the model.*", options=tone_options,
                                    index=0)    
        model_type = st.selectbox("*Select a GPT model to use for the analysis.*", options=['gpt-4o',"gpt-4-turbo","gpt-3.5-turbo-0125"],
                                    index=0)
        with st.expander("Admin Options", expanded=False):
            pwd = st.text_input('***(Admin Only)** Input password to fill API key*', type='password', value="")
            st.write('For admin password, reset chat after entering the password.')
        submit_apikey = st.form_submit_button("Submit API Key")

# Admin option to use password to fill API key
if pwd == st.secrets['admin_password']:
    st.session_state.OPENAI_API_KEY = st.secrets['MY_OPENAI_API_KEY']


##Chat Interface
st.header("🤖Ask ChatGPT")

## CREATE CHAT INTERFACE BEFORE POPULATING
## Chat Containers
menu_container = st.container(border=True)

chat_container = st.container()
output_container = chat_container.container(border=True, height=400)
user_text = chat_container.chat_input(placeholder="Enter your question here.")

ai_avatar  = "🤖"
user_avatar = "💬"


## Menu of pre-defined tasks
with menu_container:
    st.markdown("> *Select a task and click `Get Response`, or enter your own question below the chat window to get started.*")
    col1,col2 = st.columns([.6,.4])
    
    with col1:
        selected_task = st.radio("Select task:", options=task_options.keys())
        # selected_task= st.selectbox("Select task:", options=task_options.keys(), index=0)
        # model_tone = st.selectbox("*Select a tone for the model.*", options=tone_options,
        #                             index=0)        
    with col2:

        st.markdown("> *Click below to query ChatGPT*")
        get_answer_with_context = st.button("Get response.")
    
## Change User text to selected task
if get_answer_with_context:
    user_text = task_options[selected_task]


## Chat Histort Options (Download or Reset)
st.markdown("### Chat History")
sub_chat_menu = st.container(border=True)
with sub_chat_menu:
    scm_col1, scm_col2 = st.columns([.5,.5])
    
    with scm_col2:
        st.markdown("> *Click below to reset chat history.*")
        reset_button2 = st.button("Reset Chat?", key='reset 2')

reset_button1 = st.sidebar.button("Reset Chat?")

if reset_button1 or reset_button2:
    st.session_state.history = []
    # if st.session_state.OPENAI_API_KEY == "":
    #     st.error("Please enter your OpenAI API Key in the sidebar.")
    # else:
    #     st.session_state['llm'] = get_llm_no_memory(model_type=model_type,)#reset_agent(retriever=retriever)#st.session_state['retriever'] )



## Chat Output display
with output_container:
        
    # Display history
    for msg in st.session_state.history:
        if isinstance(msg, AIMessage):
            st.chat_message("assistant", avatar=ai_avatar).write(msg.content)
        
        elif isinstance(msg, HumanMessage):
            st.chat_message("user", avatar=user_avatar).write(msg.content)

    # If user input        
    if user_text is not None:
    
        ## Check for API Key
        if st.session_state.OPENAI_API_KEY == "":
            st.chat_message("SystemMessage", avatar="💻").write_stream(
                af.fake_streaming("🚨Error: Please enter your 🔑OpenAI API Key in the sidebar.")
                )
    
        else:    
            # Display User Input
            st.chat_message("user", avatar=user_avatar).write(user_text)

            # Stream response
            with st.chat_message('assistant', avatar=ai_avatar):
                response = st.write_stream(stream_response(None, 
                                                        input=user_text,
                                                        history=st.session_state.history,
                                                        resume=st.session_state.resume_text,
                                                        job = st.session_state.job_text,
                                                        model_tone=model_tone,
                                                        model_type=model_type
                                                        ))
                # ## Old non-streaming way
                # response = af.get_response(None, input=user_text,
                #                                         history=st.session_state.history,
                #                                         resume=st.session_state.resume_text,
                #                                         job = st.session_state.job_text,
                #                                         model_tone=model_tone,
                #                                         model_type=model_type)
                # st.write(af.fake_streaming(response))
            
            # Append response to history
            st.session_state.history.append(AIMessage(response))
            

## Download Chat History
scm_col1.markdown("> *Click below to download chat history.*")
scm_col1.download_button("📥Download history?", file_name="chat-history.txt", data=af.download_history())#data=json.dumps(st.session_state['history']))
