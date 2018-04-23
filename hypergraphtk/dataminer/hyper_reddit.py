import praw


class hyper_reddit:

    def __init__(self, client_id, client_secret, user_agent, username, password):
        self.reddit = praw.Reddit(client_id='XXXXXXXXXXXXXXXXXXX',
                             client_secret='XXXXXXXXXXXXXXXXXXX',
                             user_agent='XXXXXXXXXXXXXXXXXXX',
                             username='XXXXXXXXXXXXXXXXXXX',
                             password='XXXXXXXXXXXXXXXXXXX')






# assume you have a Reddit instance bound to variable `reddit`
subreddit = reddit.subreddit('redditdev')

print(subreddit.display_name)  # Output: redditdev
print(subreddit.title)         # Output: reddit Development
print(subreddit.description)   # Output: A subreddit for discussion of ...

# assume you have a Subreddit instance bound to variable `subreddit`
for submission in subreddit.new(limit=None):
    print(submission.title)  # Output: the submission's title
    print(submission.score)  # Output: the submission's score
    print(submission.id)     # Output: the submission's ID
    print(submission.url)
    print(submission.author)
    submission.comment_sort = 'new'
    top_level_comments = list(submission.comments)
    print(top_level_comments)
    # Output: the URL the submission points to
                             # or the submission's URL if it's a self post