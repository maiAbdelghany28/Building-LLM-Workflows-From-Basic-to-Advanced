# <<<< Mai Waheed AbdelMaksoud >>>>
#       <<<< 202200556 >>>>

import os
import json
import requests
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

model_server = os.getenv("MODEL_SERVER", "GROQ").upper() # Default to GROQ if not set

if model_server == "OPTOGPT":
    API_KEY = os.getenv('OPTOGPT_API_KEY')
    BASE_URL = os.getenv('OPTOGPT_BASE_URL')
    LLM_MODEL = os.getenv('OPTOGPT_MODEL')
elif model_server == "GROQ":
    API_KEY = os.getenv('GROQ_API_KEY')
    BASE_URL = os.getenv('GROQ_BASE_URL')
    LLM_MODEL = os.getenv('GROQ_MODEL')
elif model_server == "NGU":
    API_KEY = os.getenv('NGU_API_KEY')
    BASE_URL = os.getenv('NGU_BASE_URL')
    LLM_MODEL = os.getenv('NGU_MODEL')
elif model_server == "OPENAI":
    API_KEY = os.getenv('OPENAI_API_KEY')
    BASE_URL = os.getenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')  # Default to OpenAI's standard base URL
    LLM_MODEL = os.getenv('OPENAI_MODEL')
else:
    raise ValueError(f"Unsupported MODEL_SERVER: {model_server}")

client = OpenAI(api_key=API_KEY, base_url=BASE_URL)

def call_llm(messages, tools=None, tool_choice=None):
    kwargs = {
        "model": LLM_MODEL,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools
    if tool_choice:
        kwargs["tool_choice"] = tool_choice
    try:
        return client.chat.completions.create(**kwargs)
    except Exception as e:
        print(f"Error calling LLM API: {e}")
        return None


def get_sample_blog_post():
    try:
        with open("sample_blog_post.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print("Error: sample_blog_post.json file not found.")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON format in sample_blog_post.json.")
        return None
    

extract_key_points_schema = {
    "type": "function",
    "function": {
        "name": "extract_key_points",
        "description": "Extract key points from a blog post",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "The title of the blog post"
                },
                "content": {
                    "type": "string",
                    "description": "The content of the blog post"
                },
                "key_points": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "List of key points extracted from the blog post"
                }
            },
            "required": ["key_points"]
        }
    }
}

generate_summary_schema = {
    "type": "function",
    "function": {
        "name": "generate_summary",
        "description": "Generate a concise summary from the key points",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Concise summary of the blog post"
                }
            },
            "required": ["summary"]
        }
    }
}

create_social_media_posts_schema = {
    "type": "function",
    "function": {
        "name": "create_social_media_posts",
        "description": "Create social media posts for different platforms",
        "parameters": {
            "type": "object",
            "properties": {
                "twitter": {
                    "type": "string",
                    "description": "Post optimized for Twitter/X (max 280 characters)"
                },
                "linkedin": {
                    "type": "string",
                    "description": "Post optimized for LinkedIn (professional tone)"
                },
                "facebook": {
                    "type": "string",
                    "description": "Post optimized for Facebook"
                }
            },
            "required": ["twitter", "linkedin", "facebook"]
        }
    }
}

create_email_newsletter_schema = {
    "type": "function",
    "function": {
        "name": "create_email_newsletter",
        "description": "Create an email newsletter from the blog post and summary",
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": "Email subject line"
                },
                "body": {
                    "type": "string",
                    "description": "Email body content in plain text"
                }
            },
            "required": ["subject", "body"]
        }
    }
}

def task_extract_key_points(blog_post):
    messages = [
        {"role": "system", "content": "You are an expert at analyzing content and extracting key points from articles."},
        {"role": "user", "content": f"Extract key points from this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    response = call_llm(
        messages=messages,
        tools=[extract_key_points_schema],
        tool_choice={"type": "function", "function": {"name": "extract_key_points"}}
    )
    if response.choices[0].message.tool_calls:
        tool_call=response.choices[0].message.tool_calls[0]
        results = json.loads(tool_call.function.arguments)
        return results.get("key_points", [])
    return []  # Fallback if tool calling fails


def task_generate_summary(key_points,max_length=150):
    messages = [
        {"role": "system", "content": "You are an expert at summarizing content concisely while preserving key information."},
        {"role": "user", "content":  f"Generate a summary based on these key points, max {max_length} words:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[generate_summary_schema],
        tool_choice={"type": "function", "function": {"name": "generate_summary"}}
    )
    # Extract the tool call information
    if response.choices[0].message.tool_calls:
        tool_call=response.choices[0].message.tool_calls[0]
        results = json.loads(tool_call.function.arguments)
        return results.get("summary", "")
    return ""   # Fallback if tool calling fails


def task_create_social_media_posts(key_points, blog_title):
    messages = [
        {"role": "system", "content": "You are a social media expert who creates engaging posts optimized for different platforms."},
        {"role": "user", "content":  f"Create social media posts for Twitter, LinkedIn, and Facebook based on this blog title: '{blog_title}' and these key points:\n\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[create_social_media_posts_schema],
        tool_choice={"type": "function", "function": {"name": "create_social_media_posts"}}
    )
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"twitter": "", "linkedin": "", "facebook": ""}


def task_create_email_newsletter(blog_post, summary, key_points):
    messages = [
        {"role": "system", "content": "You are an email marketing specialist who creates engaging newsletters."},
        {"role": "user", "content": f"Create an email newsletter based on this blog post:\n\nTitle: {blog_post['title']}\n\nSummary: {summary}\n\nKey Points:\n" + "\n".join([f"- {point}" for point in key_points])}
    ]
    response = call_llm(
        messages=messages,
        tools=[create_email_newsletter_schema],
        tool_choice={"type": "function", "function": {"name": "create_email_newsletter"}}
    )
    if response.choices[0].message.tool_calls:
        tool_call = response.choices[0].message.tool_calls[0]
        return json.loads(tool_call.function.arguments)
    return {"subject": "", "body": ""}


def run_pipeline_workflow(blog_post):
    key_points = task_extract_key_points(blog_post)
    summary = task_generate_summary(key_points)
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    email = task_create_email_newsletter(blog_post, summary, key_points)
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }


def run_dag_workflow(blog_post):
    key_points = task_extract_key_points(blog_post)
    summary = task_generate_summary(key_points)
    social_posts = task_create_social_media_posts(key_points, blog_post['title'])
    email = task_create_email_newsletter(blog_post, summary, key_points)
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email": email
    }


def extract_key_points_with_cot(blog_post):
    """
    Extract key points from a blog post using chain-of-thought reasoning.
    Args:
        blog_post: Dictionary containing the blog post (with 'title' and 'content' keys).
    Returns:
        Dictionary with extracted key points.
    """
    # Define the system message to guide the model's reasoning process
    system_message = """
    You are an expert at analyzing content and extracting key points from articles. 
    Follow these steps to extract key points:
    1. Read the blog post carefully and identify the main topic.
    2. Break down the content into sections or themes.
    3. For each section, identify the most important ideas or arguments.
    4. Summarize each idea or argument into a concise key point.
    5. Ensure the key points are relevant, specific, and capture the essence of the blog post.
    """

    user_message = f"""
    Extract the key points from this blog post using the steps above:
    Title: {blog_post['title']}
    Content: {blog_post['content']}
    """

    # Prepare the messages for the LLM
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    response = call_llm(messages)
    if not response:
        print("Failed to call LLM for key point extraction.")
        return {}

    try:
        key_points = response.choices[0].message.content
        key_points = key_points.strip().split("\n")  # Split by \n
        return {"key_points": key_points}
    except Exception as e:
        print(f"Error parsing key points from LLM response: {e}")
        return {}


def evaluate_content(content, content_type):
    """
    Evaluate the quality of generated content using the LLM.
    Args:
        content: The content to evaluate (string or dictionary, depending on content_type).
        content_type: The type of content (e.g., "summary", "social_media_post", "email").
    Returns:
        Dictionary with evaluation results and feedback.
    """
    content_for_evaluation = content
    
    if content_type == "email":
        if isinstance(content, dict):
            content_for_evaluation = content
        else:
            content_for_evaluation = {"subject": "N/A", "body": str(content)}
    
    evaluation_criteria = {
        "summary": {
            "system_message": """
            You are an expert at evaluating summaries. Assess the quality of the summary based on:
            1. Clarity: Is the summary easy to understand?
            2. Conciseness: Is the summary brief and to the point?
            3. Completeness: Does the summary capture the key points of the original content?
            4. Relevance: Is the summary relevant to the original content?
            Provide a score out of 10 and detailed feedback.
            """,
            "user_message": f"""
            Evaluate this summary:
            {content}
            """
        },
        "social_media_post": {
            "system_message": """
            You are an expert at evaluating social media posts. Assess the quality of the post based on:
            1. Engagement: Is the post likely to attract attention and interactions?
            2. Clarity: Is the message clear and easy to understand?
            3. Relevance: Is the post relevant to the target audience?
            4. Platform Suitability: Is the post optimized for the specific platform (e.g., Twitter, LinkedIn, Facebook)?
            Provide a score out of 10 and detailed feedback.
            """,
            "user_message": f"""
            Evaluate this social media post:
            {content}
            """
        },
        "email": {
            "system_message": """
            You are an expert at evaluating email newsletters. Assess the quality of the email based on:
            1. Subject Line: Is the subject line compelling and relevant?
            2. Clarity: Is the email body clear and easy to read?
            3. Call-to-Action: Does the email include a clear and effective call-to-action?
            4. Relevance: Is the content relevant to the target audience?
            Provide a score out of 10 and detailed feedback.
            """,
            "user_message": f"""
            Evaluate this email newsletter:
            Subject: {content_for_evaluation.get('subject', 'N/A') if isinstance(content_for_evaluation, dict) else 'N/A'}
            Body: {content_for_evaluation.get('body', 'N/A') if isinstance(content_for_evaluation, dict) else str(content)}
            """
        }
    }

    if content_type not in evaluation_criteria:
        return {
            "error": f"Unsupported content type: {content_type}"
        }

    system_message = evaluation_criteria[content_type]["system_message"]
    user_message = evaluation_criteria[content_type]["user_message"]

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]

    response = call_llm(messages)
    if not response:
        return {
            "error": "Failed to call LLM for content evaluation."
        }

    try:
        evaluation_response = response.choices[0].message.content
        score = None
        feedback = evaluation_response

        if "Score:" in evaluation_response:
            score_parts = evaluation_response.split("Score:")[1].split("/")
            if len(score_parts) > 0:
                score_text = score_parts[0].strip()
                try:
                    score = float(score_text)
                except ValueError:
                    score = None
            
            feedback = evaluation_response.split("Feedback:")[1].strip() if "Feedback:" in evaluation_response else feedback

        quality_score = score / 10.0 if score is not None else 0.5

        return {
            "content_type": content_type,
            "score": score,
            "quality_score": quality_score,  
            "feedback": feedback
        }
    except Exception as e:
        return {
            "error": f"Error parsing evaluation response: {e}",
            "quality_score": 0.5  
        }


def improve_content(content, feedback, content_type):
    """
    Improve content based on feedback using the LLM.
    Args:
        content: The content to improve (string or dictionary, depending on content_type).
        feedback: Feedback on how to improve the content.
        content_type: The type of content (e.g., "summary", "social_media_post", "email").
    Returns:
        Improved content.
    """
    improvement_instructions = {
        "summary": {
            "system_message": """
            You are an expert at improving summaries. Use the feedback provided to enhance the summary.
            Ensure the improved summary is clear, concise, and captures the key points of the original content.
            """,
            "user_message": f"""
            Improve this summary based on the feedback:
            Summary: {content}
            Feedback: {feedback}
            """
        },
        "social_media_post": {
            "system_message": """
            You are an expert at improving social media posts. Use the feedback provided to enhance the post.
            Ensure the improved post is engaging, clear, and optimized for the target platform.
            """,
            "user_message": f"""
            Improve this social media post based on the feedback:
            Post: {content}
            Feedback: {feedback}
            """
        },
        "email": {
            "system_message": """
            You are an expert at improving email newsletters. Use the feedback provided to enhance the email.
            Ensure the improved email has a compelling subject line, clear body, and effective call-to-action.
            """,
            "user_message": f"""
            Improve this email newsletter based on the feedback:
            Subject: {content.get('subject', 'N/A') if isinstance(content, dict) else 'N/A'}
            Body: {content.get('body', 'N/A') if isinstance(content, dict) else str(content)}
            Feedback: {feedback}
            """
        }
    }
    if content_type not in improvement_instructions:
        return {
            "error": f"Unsupported content type: {content_type}"
        }

    system_message = improvement_instructions[content_type]["system_message"]
    user_message = improvement_instructions[content_type]["user_message"]

    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message}
    ]
    response = call_llm(messages)
    if not response:
        return {
            "error": "Failed to call LLM for content improvement."
        }

    try:
        improved_content = response.choices[0].message.content

        if content_type == "email" and isinstance(content, dict):
            subject = improved_content.split("Subject:")[1].split("\n")[0].strip() if "Subject:" in improved_content else content.get('subject', 'N/A')
            body = improved_content.split("Body:")[1].strip() if "Body:" in improved_content else content.get('body', 'N/A')
            return {
                "subject": subject,
                "body": body
            }
        else:
            return improved_content
    except Exception as e:
        return {
            "error": f"Error parsing improved content: {e}"
        }


def generate_with_reflexion(generator_func, max_attempts=3):
    """
    Apply Reflexion to a content generation function.
    Args:
        generator_func: Function that generates content
        max_attempts: Maximum number of correction attempts
    Returns:
        Function that generates self-corrected content
    """
    def wrapped_generator(*args, **kwargs):
        content_type = kwargs.pop("content_type", "summary")
       
        content = generator_func(*args, **kwargs)
        
        
        if not content:
            return content
            
        for attempt in range(max_attempts):
            evaluation = evaluate_content(content, content_type)
            
            if "error" in evaluation:
                print(f"Evaluation error: {evaluation['error']}")
                break
                
            if evaluation.get("quality_score", 0) >= 0.8:  
                return content
                
            print(f"Attempt {attempt+1}: Improving content. Current score: {evaluation.get('quality_score', 0)}")
            improved_content = improve_content(content, evaluation.get("feedback", ""), content_type)
            
            if isinstance(improved_content, dict) and "error" in improved_content:
                print(f"Improvement error: {improved_content['error']}")
                break
                
            content = improved_content
            
        return content
        
    return wrapped_generator


def run_workflow_with_reflexion(blog_post):
    """
    Run a workflow with Reflexion-based self-correction.
    Args:
        blog_post: Dictionary containing the blog post (with 'title' and 'content' keys).
    Returns:
        Dictionary with all the self-corrected content.
    """
    key_points_dict = extract_key_points_with_cot(blog_post)
    if not key_points_dict or "key_points" not in key_points_dict:
        print("Failed to extract key points.")
        return {}
    
    key_points = key_points_dict["key_points"]

    summary_generator = generate_with_reflexion(task_generate_summary, max_attempts=3)
    summary = summary_generator(key_points, content_type="summary")
    if not summary:
        print("Failed to generate summary.")
        return {}

    social_posts_generator = generate_with_reflexion(task_create_social_media_posts, max_attempts=3)
    social_posts = social_posts_generator(key_points, blog_post['title'], content_type="social_media_post")
    if not social_posts:
        print("Failed to create social media posts.")
        return {}

    email_generator = generate_with_reflexion(task_create_email_newsletter, max_attempts=3)
    email_newsletter = email_generator(blog_post, summary, key_points, content_type="email")
    if not email_newsletter:
        print("Failed to create email newsletter.")
        return {}
    return {
        "key_points": key_points,
        "summary": summary,
        "social_posts": social_posts,
        "email_newsletter": email_newsletter
    }


def define_agent_tools():
    """
    Define the tools that the workflow agent can use.
    Returns:
        List of tool definitions
    """
    all_tools = [
        extract_key_points_schema,
        generate_summary_schema,
        create_social_media_posts_schema,
        create_email_newsletter_schema
    ]
    
    finish_tool_schema = {
        "type": "function",
        "function": {
            "name": "finish",
            "description": "Complete the workflow and return the final results",
            "parameters": {
                "type": "object",
                "properties": {
                    "summary": {
                        "type": "string",
                        "description": "The final summary"
                    },
                    "social_posts": {
                        "type": "object",
                        "description": "The social media posts for each platform"
                    },
                    "email": {
                        "type": "object",
                        "description": "The email newsletter"
                    }
                },
                "required": ["summary", "social_posts", "email"]
            }
        }
    }
    
    return all_tools + [finish_tool_schema]


def execute_agent_tool(tool_name, arguments):
    """
    Execute a tool based on the tool name and arguments.
    Args:
        tool_name: The name of the tool to execute.
        arguments: The arguments to pass to the tool.
    Returns:
        The result of executing the tool.
    """
    tool_mapping = {
        "extract_key_points": task_extract_key_points,
        "generate_summary": task_generate_summary,
        "create_social_media_posts": task_create_social_media_posts,
        "create_email_newsletter": task_create_email_newsletter,
        "finish": lambda **kwargs: kwargs  
    }

    if tool_name not in tool_mapping:
        return {
            "error": f"Unsupported tool: {tool_name}"
        }

    try:
        tool_function = tool_mapping[tool_name]
        result = tool_function(**arguments)
        return result
    except Exception as e:
        return {
            "error": f"Error executing tool {tool_name}: {e}"
        }

def run_agent_workflow(blog_post):
    """
    Run an agent-driven workflow to repurpose content.
    Args:
        blog_post: Dictionary containing the blog post
    Returns:
        Dictionary with all the generated content
    """
    system_message = """
    You are a Content Repurposing Agent. Your job is to take a blog post and repurpose it into different formats:
    1. First, extract key points from the blog post using the extract_key_points tool
    2. Then, generate a concise summary using the generate_summary tool with the key points
    3. Next, create social media posts using the create_social_media_posts tool with the key points and title
    4. Finally, create an email newsletter using the create_email_newsletter tool with the blog post, summary, and key points
    5. When all tasks are complete, use the 'finish' tool to return the final results
    
    Follow this exact sequence and track your progress. Each tool call should build on the results of previous tools.
    """
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please repurpose this blog post:\n\nTitle: {blog_post['title']}\n\nContent: {blog_post['content']}"}
    ]
    
    tools = define_agent_tools()
    
    workflow_state = {
        "key_points": None,
        "summary": None,
        "social_posts": None,
        "email": None
    }
    
    max_iterations = 10
    for i in range(max_iterations):
        response = call_llm(messages, tools)
        
        if not response or not response.choices:
            return {"error": "Failed to get a response from the LLM"}
            
        messages.append(response.choices[0].message)
        if not response.choices[0].message.tool_calls:
            break
            
        for tool_call in response.choices[0].message.tool_calls:
           
            tool_name = tool_call.function.name
            arguments = json.loads(tool_call.function.arguments)
            
            if tool_name == "finish":
                final_output = {
                    "key_points": workflow_state["key_points"],
                    "summary": workflow_state["summary"] or arguments.get("summary", ""),
                    "social_posts": workflow_state["social_posts"] or arguments.get("social_posts", {}),
                    "email_newsletter": workflow_state["email"] or arguments.get("email", {})
                }
                return final_output
                
            
            tool_result = execute_agent_tool(tool_name, arguments)
            
            if tool_name == "extract_key_points" and isinstance(tool_result, list):
                workflow_state["key_points"] = tool_result
            elif tool_name == "generate_summary" and isinstance(tool_result, str):
                workflow_state["summary"] = tool_result
            elif tool_name == "create_social_media_posts" and isinstance(tool_result, dict):
                workflow_state["social_posts"] = tool_result
            elif tool_name == "create_email_newsletter" and isinstance(tool_result, dict):
                workflow_state["email"] = tool_result
                
            tool_result_str = json.dumps(tool_result) if isinstance(tool_result, (dict, list)) else str(tool_result)
            messages.append({
                "tool_call_id": tool_call.id,
                "role": "tool",
                "name": tool_name,
                "content": tool_result_str
            })
            
        if all(workflow_state.values()):
            finish_prompt = {
                "role": "user", 
                "content": "All tasks are complete. Please use the finish tool to return the final results."
            }
            messages.append(finish_prompt)
    
    if all(workflow_state.values()):
        return {
            "key_points": workflow_state["key_points"],
            "summary": workflow_state["summary"],
            "social_posts": workflow_state["social_posts"],
            "email_newsletter": workflow_state["email"]
        }
    
    return {"error": "The agent couldn't complete the workflow within the maximum number of iterations."}
def compare_workflow_approaches(blog_post):
    """
    Compare different workflow approaches for content repurposing.
    
    Args:
        blog_post: Dictionary containing the blog post (with 'title' and 'content' keys)
    
    Returns:
        Dictionary with comparative evaluation results
    """
    import time
    import json
    
    results = {}
    metrics = {}
    
    workflow_timing = {}
    
    # 1. Run Pipeline Workflow
    print("Running Pipeline Workflow...")
    start_time = time.time()
    pipeline_results = run_pipeline_workflow(blog_post)
    end_time = time.time()
    workflow_timing["pipeline"] = end_time - start_time
    results["pipeline"] = pipeline_results
    
    # 2. Run Reflexion Workflow
    print("Running Workflow with Reflexion...")
    start_time = time.time()
    reflexion_results = run_workflow_with_reflexion(blog_post)
    end_time = time.time()
    workflow_timing["reflexion"] = end_time - start_time
    results["reflexion"] = reflexion_results
    
    # 3. Run Agent-Driven Workflow
    print("Running Agent-Driven Workflow...")
    start_time = time.time()
    agent_results = run_agent_workflow(blog_post)
    end_time = time.time()
    workflow_timing["agent"] = end_time - start_time
    results["agent"] = agent_results
    
    # 4. Evaluate outputs of each approach
    print("Evaluating outputs...")
    evaluations = {}
    
    content_types = ["summary", "social_media_post", "email"]
    
    for approach, approach_results in results.items():
        evaluations[approach] = {}
        
        if "summary" in approach_results:
            evaluations[approach]["summary"] = evaluate_content(
                approach_results["summary"], 
                "summary"
            )
        
        if "social_posts" in approach_results:
            social_posts = approach_results["social_posts"]
            if "twitter" in social_posts:
                evaluations[approach]["social_media_post"] = evaluate_content(
                    social_posts["twitter"],
                    "social_media_post"
                )
        
        if "email_newsletter" in approach_results:
            evaluations[approach]["email"] = evaluate_content(
                approach_results["email_newsletter"],
                "email"
            )
    
    # 5. Calculate aggregate metrics
    for approach in results:
        metrics[approach] = {
            "avg_quality_score": 0,
            "execution_time": workflow_timing[approach],
            "content_count": 0
        }
        
        if approach in evaluations:
            quality_sum = 0
            count = 0
            
            for content_type in evaluations[approach]:
                eval_result = evaluations[approach][content_type]
                if "quality_score" in eval_result:
                    quality_sum += eval_result["quality_score"]
                    count += 1
            
            if count > 0:
                metrics[approach]["avg_quality_score"] = quality_sum / count
                metrics[approach]["content_count"] = count
    
    # 6. Perform detailed comparison analysis
    comparison = generate_comparison_analysis(results, evaluations, metrics, workflow_timing)
    
    return {
        "results": results,
        "evaluations": evaluations,
        "metrics": metrics,
        "comparison": comparison,
        "execution_times": workflow_timing
    }

def generate_comparison_analysis(results, evaluations, metrics, timing):
    """
    Generate a detailed comparison analysis of the workflow approaches.
    
    Args:
        results: Raw results from each workflow
        evaluations: Evaluation results for each output
        metrics: Aggregated metrics for each approach
        timing: Execution time for each approach
    
    Returns:
        Dictionary with detailed comparison analysis
    """
    comparison = {
        "overall_ranking": [],
        "strengths_weaknesses": {},
        "best_for": {
            "summary": None,
            "social_media": None,
            "email": None
        },
        "recommendations": []
    }
    
    ranked_approaches = sorted(
        metrics.keys(),
        key=lambda approach: metrics[approach]["avg_quality_score"],
        reverse=True
    )
    comparison["overall_ranking"] = ranked_approaches
    
    content_types = ["summary", "social_media_post", "email"]
    for content_type in content_types:
        best_approach = None
        best_score = -1
        
        for approach in evaluations:
            if content_type in evaluations[approach]:
                score = evaluations[approach][content_type].get("quality_score", 0)
                if score > best_score:
                    best_score = score
                    best_approach = approach
        
        if content_type == "social_media_post":
            comparison["best_for"]["social_media"] = best_approach
        else:
            comparison["best_for"][content_type] = best_approach
    
    comparison["strengths_weaknesses"] = {
        "pipeline": {
            "strengths": [
                "Simplest implementation",
                "Most efficient execution time" if timing["pipeline"] == min(timing.values()) else "Predictable execution flow",
                "Easy to debug and maintain"
            ],
            "weaknesses": [
                "No self-correction mechanism",
                "Fixed execution order",
                "Limited adaptability to content variations"
            ]
        },
        "reflexion": {
            "strengths": [
                "Self-improving outputs",
                "Higher quality content (especially for complex topics)",
                "Iterative refinement based on feedback"
            ],
            "weaknesses": [
                "Longer execution time",
                "More complex implementation",
                "Multiple API calls for each content piece"
            ]
        },
        "agent": {
            "strengths": [
                "Dynamic workflow planning",
                "More autonomous operation",
                "Better handling of edge cases and unexpected content",
                "Can optimize the order of operations"
            ],
            "weaknesses": [
                "Most complex implementation",
                "Less predictable execution path",
                "Potential for getting stuck in reasoning loops",
                "Higher API usage" if timing["agent"] > timing["reflexion"] else "Can be less efficient for simple content"
            ]
        }
    }
    
    best_overall = ranked_approaches[0] if ranked_approaches else None
    fastest = min(timing, key=timing.get) if timing else None
    
    comparison["recommendations"] = [
        f"For highest quality content across all types, use the {best_overall} approach",
        f"For fastest execution time, use the {fastest} approach"
    ]
    
    for content_type, approach in comparison["best_for"].items():
        if approach:
            comparison["recommendations"].append(
                f"For best {content_type.replace('_', ' ')} results, use the {approach} approach"
            )
    
    comparison["recommendations"].append(
        "For simple blog posts, the pipeline approach may be sufficient, while complex or technical content benefits more from reflexion or agent-driven approaches"
    )
    
    return comparison

def display_comparison_results(comparison_results):
    """
    Display the comparison results in a formatted way.
    
    Args:
        comparison_results: Results from compare_workflow_approaches
    """
    import json
    
    print("\n" + "="*80)
    print("WORKFLOW COMPARISON RESULTS")
    print("="*80)
    
    print("\n<<< OVERALL RANKING >>>")
    for i, approach in enumerate(comparison_results["comparison"]["overall_ranking"]):
        print(f"{i+1}. {approach.upper()} (Quality: {comparison_results['metrics'][approach]['avg_quality_score']:.2f}, Time: {comparison_results['execution_times'][approach]:.2f}s)")
    
    print("\n<<< BEST APPROACH BY CONTENT TYPE >>>")
    for content_type, approach in comparison_results["comparison"]["best_for"].items():
        if approach:
            print(f"Best for {content_type.replace('_', ' ')}: {approach.upper()}")
    
    print("\n<<< STRENGTHS & WEAKNESSES >>>")
    for approach, analysis in comparison_results["comparison"]["strengths_weaknesses"].items():
        print(f"\n{approach.upper()}:")
        print("  Strengths:")
        for strength in analysis["strengths"]:
            print(f"    - {strength}")
        print("  Weaknesses:")
        for weakness in analysis["weaknesses"]:
            print(f"    - {weakness}")
    
    print("\n<<< RECOMMENDATIONS >>>")
    for recommendation in comparison_results["comparison"]["recommendations"]:
        print(f"- {recommendation}")
    
    # Display detailed metrics
    print("\n<<< DETAILED METRICS >>>")
    for approach, metrics in comparison_results["metrics"].items():
        print(f"\n{approach.upper()}")
        print(f"  Average Quality Score: {metrics['avg_quality_score']:.2f}")
        print(f"  Execution Time: {metrics['execution_time']:.2f} seconds")
        print(f"  Content Count: {metrics['content_count']}")
    
    print("\n" + "="*80)

def main_with_comparison():
    """
    Run the content repurposing workflow with comparative evaluation.
    """
    # Load a sample blog post
    blog_post = get_sample_blog_post()
    if not blog_post:
        print("Error: Unable to load the sample blog post.")
        return

    print("Running comparative evaluation of content repurposing workflows...\n")

    # Run comparison
    comparison_results = compare_workflow_approaches(blog_post)
    
    # Display results
    display_comparison_results(comparison_results)
    
    # Option to save detailed results to a file
    save_option = input("\nWould you like to save detailed results to a file? (y/n): ")
    if save_option.lower() == 'y':
        filename = "workflow_comparison_results.json"
        with open(filename, 'w') as f:
            json.dump(comparison_results, f, indent=2)
        print(f"Detailed results saved to {filename}")
    
    print("\nComparative evaluation complete.")

def main():
    """
    Main function to run the content repurposing workflow.
    """
    # Load a sample blog post (for demonstration purposes)
    blog_post = get_sample_blog_post()
    if not blog_post:
        print("Error: Unable to load the sample blog post.")
        return

    print("Running the content repurposing workflow...\n")

    # Run the pipeline workflow
    print("=== Running Pipeline Workflow ===")
    pipeline_results = run_pipeline_workflow(blog_post)
    if pipeline_results:
        print("\nPipeline Workflow Results:")
        print(json.dumps(pipeline_results, indent=2))
    else:
        print("Pipeline workflow failed.")

    # Run the DAG workflow
    print("\n=== Running DAG Workflow ===")
    dag_results = run_dag_workflow(blog_post)
    if dag_results:
        print("\nDAG Workflow Results:")
        print(json.dumps(dag_results, indent=2))
    else:
        print("DAG workflow failed.")

    # Run the workflow with Reflexion
    print("\n=== Running Workflow with Reflexion ===")
    reflexion_results = run_workflow_with_reflexion(blog_post)
    if reflexion_results:
        print("\nReflexion Workflow Results:")
        print(json.dumps(reflexion_results, indent=2))
    else:
        print("Reflexion workflow failed.")

    # Run the agent-driven workflow
    print("\n=== Running Agent-Driven Workflow ===")
    agent_results = run_agent_workflow(blog_post)
    if agent_results:
        print("\nAgent-Driven Workflow Results:")
        print(json.dumps(agent_results, indent=2))
    else:
        print("Agent-driven workflow failed.")

    print("\nWorkflow execution complete.")


if __name__ == "__main__":
    main()
    main_with_comparison()