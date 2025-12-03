简介：
  一个模拟vllm流式响应的代码。
	如果手边暂时没有能运行vllm服务的条件，但希望在修改前端程序的时候进行简单的响应测试，可以启动这个程序。
  它将正常接受来自API服务的输入并以固定内容进行流式输出，代替vllm的效果。
	不依赖cuda环境和vllm库，可以直接在普通核显windows powershell中运行。
  
测试通过环境：
  Windows 11 25H2 Python3.12
  Ubuntu 24.04 Python3.12

启动方法：
	打开Windows PowerShell 或者 Ubuntu 终端
	进入项目目录后：
```bash
	cd mock_vllm
	python start_mock_vllmserver.py
```
然后另开两个终端，分别启动 API服务 和 前端服务（如cli），即可正常测试。

Intro:
	Code for simulating vLLM's streaming response.
	If you do not have the conditions to run a vLLM service for the time being but need to perform simple response testing	while modifying the frontend program, you can launch this program. 
  It will normally accept input from the API service 	and output fixed content in a streaming manner, acting as a replacement for the vLLM service.
	It has no dependencies on a CUDA environment or the vLLM library, and can be run directly in PowerShell on a regular 	Windows system with integrated graphics

Tested Environments:
  Windows 11 25H2 with Python 3.12
  Ubuntu 24.04 with Python 3.12

Startup Method:
	Open Windows PowerShell or the Ubuntu terminal.
	After navigating to the project directory:
```bash
	cd mock_vllm
	python start_mock_vllmserver.py
```
	Then open two additional terminals, start the API service and the frontend service (e.g., CLI) respectively, and you can 	proceed with normal testing.



  
