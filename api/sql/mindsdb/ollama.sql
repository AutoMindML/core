-- ml engine
create ml_engine ollama_engine
from ollama;

-- model from ml engine
create model llama3_model
predict completion
using
    engine = 'ollama_engine',
    model_name = 'llama3.2',
    ollama_server_url = 'http://localhost:11434';

-- check info
describe llama3_model;

-- test model
select text, completion
from llama3_model
where text = 'Hello';
