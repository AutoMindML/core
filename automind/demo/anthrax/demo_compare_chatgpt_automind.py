import sys

from demo_anthrax_automind import testing as automind_test
from demo_anthrax_automind import validation as automind_validation
from demo_anthrax_chatgpt import testing as chatgpt_test
from demo_anthrax_chatgpt import validation as chatgpt_validation
from demo_anthrax_chatgpt_advance import testing as chatgpt_advance_test
from demo_anthrax_chatgpt_advance import validation as chatgpt_advance_validation

if __name__ == "__main__":
    testing = False

    if len(sys.argv) > 1:
        arg1 = sys.argv[1]

        if arg1 == "test":
            testing = True

    if not testing:
        chatgpt_validation()
        chatgpt_advance_validation()
        automind_validation()
    else:
        chatgpt_test()
        chatgpt_advance_test()
        automind_test()
