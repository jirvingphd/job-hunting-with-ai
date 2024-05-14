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
from langchain.memory import ConversationBufferWindowMemory
import os
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser


st.set_page_config(page_title="Resume and Job Listing Analyzer", page_icon="💼")#, layout="wide")

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
    # pdf_file_obj.close()
    return text

## Initialize session state
if 'resume_file' not in st.session_state: 
    st.session_state.resume_file = None

if 'resume_text' not in st.session_state:
    st.session_state.resume_text = None
    
if 'job_listing_file' not in st.session_state:    
    st.session_state.job_listing_file = None

if "history" not in st.session_state:
    st.session_state['history'] = []

if 'job_text' not in st.session_state:
    st.session_state.job_text = None

if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state.OPENAI_API_KEY = ""


st.title('Resume and Job Listing Analyzer')

# c1, c2 = st.columns([.3,.6])
st.sidebar.image("images/DALLE-2.png", use_column_width=True)
st.sidebar.markdown("*This app uses ChatGPT to analyze your resume and a job listing to provide tailored advice and recommendations.*")
### HERE
with st.sidebar.container(border=True):
    st.subheader("OpenAI API Key")
    st.write('> *Enter your OpenAI API key below. You can sign up for one [here](https://platform.openai.com/api-keys).*')
    st.session_state['OPENAI_API_KEY']  = st.text_input("OpenAI API Key", type="password", value=st.session_state.OPENAI_API_KEY)
    # with st.sidebar.expander("GPT Model"):
    # st.write('>*Select a GPT model to use for the analysis.*')
    model_type = st.selectbox("*Select a GPT model to use for the analysis.*", options=['gpt-4o',"gpt-4-turbo","gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct",],
                                index=0)
    pwd = st.text_input('(Admin Only) Input password to fill API key', type='password')



if 'OPENAI_API_KEY' not in st.session_state:
    st.session_state['OPENAI_API_KEY'] = ""

if pwd == 'formerninja':
    try:
        st.session_state['OPENAI_API_KEY'] = os.getenv ('OPENAI_API_KEY')
    except:
        st.session_state['OPENAI_API_KEY'] = st.secrets['OPENAI_API_KEY']
# else:
#         st.session_state['OPENAI_API_KEY'] = ""#os.getenv("OPENAI_API_KEY")

with st.container(border=True):
    st.markdown("- 👈 First, use the sidebar (`>`) to upload your OPENAI Api Key.")
    st.markdown('- 👇*Next, upload or paste your  resume and job listing documents.*')


# st.divider()

docs_container = st.expander("### Upload Documents", expanded=True)#border=True)
with docs_container:
    # st.markdown('#### Upload Documents')# resume and the job listing to get started')
    # st.sidebar.header("Resume")
    c1, c2 = st.columns([.5,.5])
    resume_container   =    c1.container(border=True)#st.sidebar.container(border=True)
    resume_container.markdown("#### Resume")
    # st.sidebar.header("Job Listing")
    job_listing_container = c2.container(border=True)#st.sidebar.container(border=True)
    job_listing_container.markdown("#### Job Listing")
# st.sidebar.divider()

# st.divider()

## Upload pdf or paste resume
with resume_container:
    resume_form =st.form(key='resume_form', border=False)
    
    with resume_form:
        st.session_state.resume_file = st.file_uploader("Upload your PDF resume", type="pdf", accept_multiple_files=False)
    
        # Pasted Version
        st.session_state.pasted_resume = st.text_area("or paste your resume here:")
        submit_resume = st.form_submit_button("Update resume.")
    # st.session_state.pasted_resume = st.text_area("or paste your resume here:", height=100,)



## Upload pdf or past job listing    
with job_listing_container:
    job_form =st.form(key='job_form', border=False)

    with job_form:
        
        st.session_state.job_listing_file = st.file_uploader("Upload the PDF job listing", type="pdf", accept_multiple_files=False)
        # Pasted version
        st.session_state.pasted_job_listing = st.text_area("Paste the job listing here", height=100)
        submit_job = st.form_submit_button("Update job listing.")
        




## Set resume text
def get_resume_text():
    
    if st.session_state.pasted_resume != '':
        st.session_state.resume_text = st.session_state.pasted_resume
    elif st.session_state.resume_file is not None:
        st.session_state.resume_text = read_pdf(st.session_state.resume_file) 
    else:
        st.error("Please upload a resume or paste it in the text area.")
        
if submit_resume:
    get_resume_text()
    # if st.session_state.pasted_resume is not None:
    #     st.session_state.resume_text = st.session_state.pasted_resume
    # elif st.session_state.resume_file is not None:
    #     st.session_state.resume_text = read_pdf(st.session_state.resume_file) 
    # else:
    #     st.error("Please upload a resume or paste it in the text area.")
    # else:  
        # st.session_state.resume_text = None


## set job listing text
def get_job_text():
    if st.session_state.pasted_job_listing != '':
        st.session_state.job_text = st.session_state.pasted_job_listing
    elif st.session_state.job_listing_file is not None:
        st.session_state.job_text = read_pdf(st.session_state.job_listing_file)
    else:
        st.error("Please upload a job listing or paste it in the text area.")
        # st.session_state.job_text = None
if submit_job:
    get_job_text()
    # if st.session_state.job_listing_file:
    #     st.session_state.job_text = read_pdf(st.session_state.job_listing_file)
    # elif st.session_state.pasted_job_listing:
    #     st.session_state.job_text = st.session_state.pasted_job_listing
    # else:
    #     st.error("Please upload a job listing or paste it in the text area.")
    #     # st.session_state.job_text = None
# else:
#     st.session_state.job_text = ''



def get_template_string():
    
    return """You are a a specialized career coach for the data science and analytics sector, focused on delivering tailored, concise job application advice. 
    You are proficient in resume analysis, cover letter guidance, and interview preparation, adapting to each user's unique requirements.
    When analyzing a resume vs. a job listing, start by categorizing a user's fit for a job as 'perfect,' 'great,' 'good,' or 'non-ideal' based on the resume and job listing comparison before going into detail.
    You maintain a professional, friendly tone, and encouraging tone, ensuring advice is efficient, clear, and easily understandable, with the goal of enhancing user confidence and aiding their career progression.
    """


def get_system_prompt_str():
    """Helper function for get_prompt_template. New v2.0"""
    system_prompt = (" You are a a specialized career coach for the {sector}, focused on delivering tailored, concise job application advice and practice. " 
    " You are proficient in resume analysis, cover letter guidance, and interview preparation, adapting to each user's unique requirements. "
    " You maintain a professional, friendly tone, and encouraging tone, ensuring advice is efficient, clear, and easily understandable, "
    " with the goal of aiding their career progression. Ask the user for their resume and job listing if not provided and they are needed to asnwer .")
    context = "\nUse the following context, if provided, to help answer the questions:\n\nHere is my resume:\n-------------\n {resume}\n\n Here is the job listing:\n-------------\n{job}\n\n "    
    return system_prompt + context


def get_llm_no_memory(model_type="gpt-3.5-turbo-0125", temperature=0.1, #
            system_prompt_template_func= get_system_prompt_str,#verbose=False,
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
    final_prompt_template = final_prompt_template.partial(sector=sector)#, resume=resume, job=job)
        
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
        
            

def get_response(llm_no_mem, input, resume='', job='', history=[]):
    output = llm_no_mem.invoke({'input':input,
                   'resume':resume,
                   'job':job,
                   'history':history,})

    # response = llm_no_mem.invoke({"resume":resume,
    #                               "job":job,
    #                               'input':input_text,
    #                               'history':history})
    
    if isinstance(output, dict):
        response = output['response']
    else:
        response = output
    history.append(HumanMessage(content=input))
    history.append(AIMessage(response))
    return response, history
    
    
        
## For steramlit try this as raw code, not a function
def print_history(llm_chain):
    # Simulate streaming for final message
    if isinstance(llm_chain, ConversationChain):
        session_state_messages = llm_chain.memory.buffer_as_messages
    elif isinstance(llm_chain, list):
        session_state_messages = llm_chain
    else:
        session_state_messages=[]
        
    for msg in session_state_messages:
        if isinstance(msg, AIMessage):
            # notebook
            # streamlit
            st.chat_message("assistant", avatar=ai_avatar).write(msg.content)
        
        elif isinstance(msg, HumanMessage):
            # notebook
            # print(f"User: {msg.content}")
            # streamlit
            st.chat_message("user", avatar=user_avatar).write(msg.content)
        print()



def get_task_options(prompt_config_file = "config/prompt_config.json" ,options_only=False):
    with open(prompt_config_file, 'r') as f:
        task_prompt_dict = json.load(f)
    if options_only:
        return list(task_prompt_dict.keys())
    else:
        return task_prompt_dict

# st.divider()
st.header("Ask ChatGPT")
# st.header("AI Recommendations")
# summary_container = st.container()
menu_container = st.container(border=True)

chat_container = st.container()
# chat_container.header("Q&A")
output_container = chat_container.container(border=True, height=400)
user_text = chat_container.chat_input(placeholder="Enter your question here.")

ai_avatar  = "🤖"
user_avatar = "💬"


## CHANGE CODE TO ADD HUMAN/AI MESSAGE to empty history
# Specify task options
# task_options = get_task_options(options_only=True) #task_prompt_dict.keys()
task_options  = get_task_options(options_only=False)


with menu_container:
    ## Select summary or recommendation
    # col1,col2,col3 = st.columns([.4,.3,.3])#3)
    col1,col2 = st.columns([.6,.4])
    # show_summary = col1.button("Show Summary of Customer Sentiment")
    # show_recommendations = col1.button("Get product improvement recommendations",)
    # show_marketing_recs = col2.button("Get marketing recommendations.")
    selected_task = col1.radio("Select task:", options=task_options.keys())
    col2.markdown("> *Click below to query ChatGPT*")
    get_answer_with_context = col2.button("Get response.")

    # reset_button2 = col3.button("Reset Chat?", key='reset 2')

st.divider()
sub_chat_menu = st.container(border=True)
with sub_chat_menu:
    scm_col1, scm_col2 = st.columns([.5,.5])
    scm_col2.markdown("> *Click below to reset chat history.*")

    reset_button2 = scm_col2.button("Reset Chat?", key='reset 2')

reset_button1 = st.sidebar.button("Reset Chat?")
if ('llm' not in st.session_state) or reset_button1 or reset_button2:
    # agent = get_agent(retriever)
    # with output_container:
    st.session_state['history'] = []
    # get_job_text()
    # get_resume_text()
    st.session_state['llm'] = get_llm_no_memory(model_type=model_type,)#reset_agent(retriever=retriever)#st.session_state['retriever'] )


# with chat_container:
# output_container = st.container(border=True)
with output_container:
        
    print_history(st.session_state['history'])#st.session_state['llm'])
    if get_answer_with_context:
        user_text = task_options[selected_task]
        
    if user_text is not None:
        st.chat_message("user", avatar=user_avatar).write(user_text)

        response, history = get_response(st.session_state['llm'], input=user_text,
                                         resume=st.session_state.resume_text,
                                         job = st.session_state.job_text,
                                         history=st.session_state['history'])
        st.session_state['history']  = history
        # response = st.session_state['llm'].invoke({"input":user_text})
        st.chat_message('assistant', avatar=ai_avatar).write(fake_streaming(response))#response['response']))


    
def download_history():
        
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
    return "\n\n".join(md_history)
    # if submit_button:
    
# chat_options.markdown("**Clear conversation history.**")

# reset_button  = chat_options.button("Clear",on_click=reset)
scm_col1.markdown("> *Click below to download chat history.*")
scm_col1.download_button("Download history?", file_name="chat-history.txt", data=download_history())#data=json.dumps(st.session_state['history']))
