from models.summarization import summarize_email
from models.ollama import summarize_email_llama

email_content = """
    Subject: Just Checking In – Any Thoughts on Our Offer?

    Hi John,

    I hope you’re doing well! I wanted to quickly follow up on the proposal we sent over last week for the Acme Sales Solution. I know you’ve been busy, but I wanted to see if you had any questions or concerns that we can help address.

    I’m confident that our solution can really streamline your current sales process, especially with the new integrations we’ve built in for CRM and reporting. Plus, the team is excited about the opportunity to work with you guys, and I’m sure we can put together a custom package that fits within your budget.

    If you’re ready, we could set up a quick call this week to go over any final details and figure out the next steps. We’re also running a special discount this month, so it might be a great time to lock everything in!

    Let me know what works best for you. Looking forward to hearing your thoughts.

    Best regards,
    Mike Carter
    Account Executive
    Acme Solutions
"""

k = """
    Subject: Following Up on Our Product Demo – Next Steps

    Hi Sarah,

    I hope you’re doing well! I wanted to follow up on our recent demo of the Acme Enterprise Platform and see if you had any further thoughts or questions. We’re really excited about the opportunity to work with your team and believe our platform can make a significant impact on streamlining your processes.

    From our last conversation, it sounded like cost reduction and scalability were two of the biggest areas you’re focusing on right now. Just to recap:

        •	Cost Reduction: Our solution can help you reduce operational costs by up to 20% through automation, and we offer flexible pricing options based on the features you actually need.
        •	Scalability: As your business grows, our platform can easily scale with you, so you won’t have to worry about infrastructure costs down the line.

    I also wanted to address one of the questions that came up during the demo about data security. We take data security very seriously, and I’ve attached some additional details on our end-to-end encryption and compliance certifications to make sure that aligns with what your team is looking for.

    If you’d like to move forward, I’d love to schedule a quick call later this week to talk through any final questions or concerns. We’re offering a special discount for new clients this quarter, and I’d be happy to walk you through that as well.

    Looking forward to hearing your thoughts!

    Best,
    Mark Thompson
    Senior Sales Representative
    Acme Solutions

"""

# Call summarization
summary = summarize_email_llama(email_content)
print(f"Summary: {summary}")


