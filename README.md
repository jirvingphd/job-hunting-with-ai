# job-hunting-with-ai
 
- Streamlit App: https://job-hunting-with-ai.streamlit.app/

<img src="images/app-snapshot.png" style="border:solid 1px black">

___
JMI Notes re: venv

```bash
conda deactivate
python -m venv app-env
source app-env/bin/activate
pip install -r requirements.txt
```
___

## TO DO (04/15/24)
- Continue working in app-testing.ipynb to:
    - [ ] Re-send the resume and job fields for the chat template every time a message is sent. 
    -  [ ] Use an external list for memory
    
- Once workflow is figured out, update app.py:
    - [ ] Update logic for llm and responses from notebook above.
    - [ ] Write code for app to select the correct text using teh sidebar forms for resume and job listing.