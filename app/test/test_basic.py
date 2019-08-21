import os
import sys
print ("os.getcwd() -> ",os.getcwd())
sys.path.append(os.getcwd())

from nose.tools import eq_, ok_
import unittest 
import json
import run_server

class BasicTests(unittest.TestCase):
 
    ############################
    #### setup and teardown ####
    ############################
 
    # executed prior to each test
    def setUp(self):
        self.app = run_server.app.test_client()
    
        # executed after each test
    def tearDown(self):
        pass
 
    ###############
    #### tests ####
    ###############
 
    def test_hello(self):
        res = self.app.get("/")
        eq_(200, res.status_code)

    def test_model_info_before_init(self):
        res = self.app.get("/model")
        eq_(404, res.status_code)

    def test_model_init(self):
        res = self.app.get("/model/init")        
        eq_(200, res.status_code)
 
    def test_model_info_after_init(self):
        res = self.app.get("/model")
        eq_(404, res.status_code)

if __name__ == "__main__":
    unittest.main()