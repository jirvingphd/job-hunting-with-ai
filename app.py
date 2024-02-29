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

if ['resume_file','job_listing_file'] not in st.session_state:
    
    st.session_state.resume_file = None
    st.session_state.resume_text = None
    
    st.session_state.job_listing_file = None
    st.session_state.job_listing_text = None

st.title('Resume and Job Listing Analyzer')

# c1, c2 = st.columns([.3,.6])
st.sidebar.image("images/DALLE-2.png", use_column_width=True)
st.sidebar.markdown("*This app uses OpenAI's GPT-3.5 model to analyze your resume and a job listing to provide tailored advice and recommendations.*")
with st.container(border=True):
    st.markdown(">- 👈 Use the sidebar (`>`) to upload your resume and the job listing as PDFs to get started.")



# st.divider()

st.sidebar.header("Upload Files")
# st.sidebar.markdown('> Upload your resume and the job listing to get started')
st.session_state.resume_file = st.sidebar.file_uploader("Upload your PDF resume", type="pdf", accept_multiple_files=False)
st.session_state.job_listing_file = st.sidebar.file_uploader("Upload the PDF job listing", type="pdf", accept_multiple_files=False)

if st.session_state.resume_file:
    # st.write([i for i in dir(resume_file) if not i.startswith('_')])
    st.session_state.resume_text = read_pdf(st.session_state.resume_file)

    
if st.session_state.job_listing_file:
    st.session_state.job_listing_text = read_pdf(st.session_state.job_listing_file)




def get_template_string():
    
    return """You are a a specialized career coach for the data science and analytics sector, focused on delivering tailored, concise job application advice. 
    You are proficient in resume analysis, cover letter guidance, and interview preparation, adapting to each user's unique requirements.
    When analyzing a resume vs. a job listing, start by categorizing a user's fit for a job as 'perfect,' 'great,' 'good,' or 'non-ideal' based on the resume and job listing comparison before going into detail.
    You maintain a professional, friendly tone, and encouraging tone, ensuring advice is efficient, clear, and easily understandable, with the goal of enhancing user confidence and aiding their career progression.
    """

st.sidebar.divider()
st.sidebar.header("GPT Model")
model_type = st.sidebar.radio("GPT Model", options=["gpt-3.5-turbo-0125", "gpt-3.5-turbo-instruct"],#,"gpt-4-0125-preview"],
                              index=0)

def get_llm(model_type=model_type, temperature=0.1, #verbose=False,
             template_string_func=get_template_string):

    ## get prompt string
    system_prompt = template_string_func()
    
    prompt_template =system_prompt+"""
    Current conversation:
    {history}
    Human: {input}
    AI:"""
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(temperature=.1, model=model_type)
    llm_chain = ConversationChain(prompt=prompt, llm=llm, memory=ConversationBufferWindowMemory(),#ConversationBufferMemory(memory_key="history"),
                                  verbose=True, output_key="response")#,#callbacks=callbacks)
    
    # llm_chain.memory.chat_memory.add_sy (template)
    return llm_chain


            
            
# def reset_agent(#fpath_db = FPATHS['data']['app']['vector-db_dir'],
#                 # retriever=retriever, #st.session_state['retriever'] , 
#                 starter_message = "Hello, there! Enter your question here and I will check the full reviews database to provide you the best answer.",
#                get_agent_kws={}):
#     # fpath_db
#     agent_exec = get_llm( **get_agent_kws)
#     agent_exec.memory.chat_memory.add_ai_message(starter_message)
#     with chat_container:
#         st.chat_message("assistant", avatar=ai_avatar).write_stream(fake_streaming(starter_message))
#         # print_history(agent_exec)
#     return agent_exec
    

def fake_streaming(response):
    import time
    for word in response.split(" "):
        yield word + " "
        time.sleep(.05)		
        
            
    
## For steramlit try this as raw code, not a function
def print_history(llm_chain):
    # Simulate streaming for final message

    session_state_messages = llm_chain.memory.buffer_as_messages
    for msg in session_state_messages:#[:-1]:
        if isinstance(msg, AIMessage):
            # notebook
            print(f"Assistant: {msg.content}")
            # streamlit
            st.chat_message("assistant", avatar=ai_avatar).write(msg.content)
        
        elif isinstance(msg, HumanMessage):
            # notebook
            print(f"User: {msg.content}")
            # streamlit
            st.chat_message("user", avatar=user_avatar).write(msg.content)
        print()




def get_context_strings(context_resume=None, context_job=None,):
    if context_resume is None:
        context_resume = st.session_state.resume_text
    
    if context_job is None:
        context_job = st.session_state.job_listing_text
    # task_prompt_dict = get_task_options(options_only=False)
    # system_prompt = task_prompt_dict[selected_task]
    
    # template_assistant = "You are a helpful assistant data scientist who uses NLP analysis of customer reviews to inform business-decision-making:"
    # product_template = f" Assume all user questions are asking about the content in the user reviews. Note the product metadata is:\n```{product_string}```\n\n"
    # template_starter = get_template_string()
    context = f"\nHere is my resume: \n\n-------------\n {context_resume}.\n\n Here is the job listing:\n\n-------------\n{context_job}.\n\n"
    # context += f"Use the {context_type} first before using the retrieved documents."
    # template_assistant=template_starter+ context
    return context


def get_task_options(options_only=False):
    task_prompt_dict= {
        # "Summary of Customer Sentiment":'Provide a summary list of what 1-star reviews did not like and a summary of what did 5-star reviews liked.',
                   'Compare Resume vs. Job':'How does my resume compare against this job listing? Provide a detailed break down list comparing my resume vs. the listing. Highlight any weaknesses.',
                   'Craft Covert Letter':'Write a cover letter for this job. Use a non-traditional opening sentence that is very enthusiastic about the job and memorable (you can even use !). The reamining portions of the letter should be high quality and speak to my experience.',
                   "Interview Prep":'Help me prepare for an interview. Start with a question that I should expect in the interview.',
                   "Resume Analysis":'Analyze my resume. Start with a summary of my resume.',}

    if options_only:
        return list(task_prompt_dict.keys())
    else:
        return task_prompt_dict


# st.header("AI Recommendations")
# summary_container = st.container()
menu_container = st.container(border=True)

chat_container = st.container()
# chat_container.header("Q&A")
output_container = chat_container.container(border=True)
user_text = chat_container.chat_input(placeholder="Enter your question here.")

ai_avatar  = "🤖"
user_avatar = "💬"


## CHANGE CODE TO ADD HUMAN/AI MESSAGE to empty history
# Specify task options
# task_options = get_task_options(options_only=True) #task_prompt_dict.keys()
task_options  = get_task_options(options_only=False)


with menu_container:
    ## Select summary or recommendation
    col1,col2,col3 = st.columns([.4,.3,.3])#3)
    # show_summary = col1.button("Show Summary of Customer Sentiment")
    # show_recommendations = col1.button("Get product improvement recommendations",)
    # show_marketing_recs = col2.button("Get marketing recommendations.")
    selected_task = col1.radio("Select task:", options=task_options.keys())
    col2.markdown("> *Click below to query ChatGPT*")
    get_answer_with_context = col2.button("Get response.")
    col3.markdown("> *Click below to reset chat history.*")

    reset_button2 = col3.button("Reset Chat?", key='reset 2')


reset_button1 = st.sidebar.button("Reset Chat?")
if ('llm' not in st.session_state) or reset_button1 or reset_button2:
    # agent = get_agent(retriever)
    with output_container:
        st.session_state['llm'] = get_llm()#reset_agent(retriever=retriever)#st.session_state['retriever'] )


# with chat_container:
# output_container = st.container(border=True)
with output_container:
        
    print_history(st.session_state['llm'])
    if user_text:
        st.chat_message("user", avatar=user_avatar).write(user_text)
    
        response = st.session_state['llm'].invoke({"input":user_text})
        st.chat_message('assistant', avatar=ai_avatar).write(fake_streaming(response['response']))

    if get_answer_with_context:
        prompt_text =  task_options[selected_task]
        context = get_context_strings()
        st.chat_message("user", avatar=user_avatar).write(prompt_text)
        
        response = st.session_state['llm'].invoke({'input':prompt_text+context})

        
        # print_history(st.session_state['agent-summarize'])

        # response = st.session_state['agent'].invoke({"input":prompt_text})
        st.chat_message('assistant', avatar=ai_avatar).write(fake_streaming(response['response']))


        # print_history(st.session_state['llm'])
    
def download_history():
        
    avatar_dict = {'human': user_avatar,
                   'ai':ai_avatar,
                   'SystemMessage':"💻"}
    
    md_history = []
    history = st.session_state['llm'].memory.buffer_as_messages
    for msg in history:
        type_message = msg.type#type(msg) x
            # with st.chat_message(name=i["role"],avatar=avatar_dict[ i['role']]):
        md_history.append(f"{avatar_dict[type_message]}: {msg.content}")
    return "\n\n".join(md_history)
    # if submit_button:
    
# chat_options.markdown("**Clear conversation history.**")

# reset_button  = chat_options.button("Clear",on_click=reset)
st.download_button("Download history?", file_name="chat-history.txt", data=download_history())#data=json.dumps(st.session_state['history']))

    
# if uploaded_file is not None:
#      # To read file as bytes:
#      bytes_data = uploaded_file.getvalue()
#      st.write(bytes_data)
     
#      # To convert to a string based IO:
#     #  stringio = StringIO(uploaded_file.getvalue().decode("utf-8"))
#     #  st.write(stringio
              
#      # To read file as string:
#      string_data = stringio.read()
#      st.write(string_data)