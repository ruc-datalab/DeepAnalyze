import sys
import signal
import subprocess
import asyncio
from server import jupyter_process, bot_stream

def signal_handler(sig, frame):
    """Process Ctrl+C signal to stop terminal and Jupyter process"""
    print("\nStopping terminal and Jupyter Lab server...")
    if jupyter_process:
        jupyter_process.terminate()
        try:
            jupyter_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            jupyter_process.kill()
    sys.exit(0)

async def interactive_chat():
    """Interactive chat function to handle user input and bot responses"""
    print("Interactive chat started. Type your messages and press Enter to send.")
    print("Type 'exit', 'quit' or press Ctrl+C to exit.")
    print("=" * 50)
    
    messages = []
    
    while True:
        try:
            user_input = input(">>> ")
            if user_input.lower() in ['exit', 'quit']:
                print("Exiting chat...")
                break
                
            if not user_input.strip():
                continue
                
            # Add user message to the conversation
            messages.append({"role": "user", "content": user_input})
            messages = await bot_stream(messages)
            print("Successfully received bot response.")
            # print(messages)
            print("-" * 50)
            
        except EOFError:
            print("\nDetected EOF, exiting chat...")
            break
        except KeyboardInterrupt:
            print("\nKeyboard interrupt detected, exiting chat...")
            break
        except Exception as e:
            print(f"\nError occurred: {e}")
            break

if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    
    # Run the interactive chat
    try:
        asyncio.run(interactive_chat())
    except KeyboardInterrupt:
        pass
    finally:
        # Stop Jupyter process if it's still running
        if jupyter_process:
            print("Stopping Jupyter Lab server...")
            jupyter_process.terminate()
            try:
                jupyter_process.wait(timeout=5)
            except:
                jupyter_process.kill()