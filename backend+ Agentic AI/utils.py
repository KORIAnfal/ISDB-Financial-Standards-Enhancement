import time
import re
import os

# Import specific RateLimitError types from the main app context or handle generically
# Keeping the import in app.py and passing the tuple is safer
# from groq import RateLimitError as GroqRateLimitError
# from openai import RateLimitError as OpenAIRateLimitError
# RateLimitError = (GroqRateLimitError, OpenAIRateLimitError)


def invoke_agent_with_retry(agent_chain, inputs, max_retries, initial_delay, RateLimitError):
    """Invokes an LLMChain agent with retry logic for RateLimitError."""
    delay = initial_delay
    for i in range(max_retries):
        try:
            return agent_chain.invoke(inputs)
        except RateLimitError as e:
            print(f"Rate limit error encountered (Attempt {i+1}/{max_retries}): {e}")
            if i < max_retries - 1:
                # Check if error message contains specific retry-after header info
                retry_after_match = re.search(r'Please try again in ([\d.]+)s', str(e))
                if retry_after_match:
                    suggested_delay = float(retry_after_match.group(1))
                    # Use the suggested delay if it's larger than our current delay
                    delay = max(delay, suggested_delay)
                    print(f"API suggested retry after {suggested_delay:.2f}s. Using delay {delay:.2f}s.")
                else:
                     print(f"No specific retry-after header found. Using default delay {delay:.2f}s.")

                time.sleep(delay)
                delay *= 2 # Exponential backoff
            else:
                print(f"Max retries ({max_retries}) reached. Failing.")
                raise # Re-raise the last exception
        except Exception as e:
            # Catch other potential errors during invoke
            print(f"An unexpected error occurred during agent invocation (Attempt {i+1}/{max_retries}): {e}")
            # For other errors, we don't necessarily want to retry unless it's a transient connection issue
            # For simplicity, we'll just re-raise them.
            raise

def load_standard_from_file(file_path):
    """Loads standard text from a file."""
    try:
        # Adjust path assuming utils.py is in the same directory as app.py
        # and the standards directory is a sibling.
        base_dir = os.path.dirname(os.path.abspath(__file__))
        full_path = os.path.join(base_dir, file_path)
        with open(full_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        print(f"ERROR: File not found at {full_path}")
        return None
    except Exception as e:
        print(f"ERROR: Could not read file {full_path}: {e}")
        return None