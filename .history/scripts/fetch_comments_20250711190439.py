from googleapiclient.discovery import build
import pandas as pd

# Replace with your API key
api_key = ""

# Replace with your YouTube video ID (found in video URL after v=)
video_id = "VIDEO_ID_HERE"

# Build YouTube service
youtube = build("youtube", "v3", developerKey=api_key)


def get_comments(youtube, video_id):
    comments = []
    request = youtube.commentThreads().list(
        part="snippet", videoId=video_id, maxResults=100, textFormat="plainText"
    )
    response = request.execute()

    while request:
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
            comments.append(comment)

        if "nextPageToken" in response:
            request = youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=response["nextPageToken"],
                textFormat="plainText",
            )
            response = request.execute()
        else:
            break
    return comments


# Fetch comments
comments_list = get_comments(youtube, video_id)

# Convert to DataFrame
df = pd.DataFrame(comments_list, columns=["comment"])

# Save to CSV in data folder
df.to_csv("../data/youtube_comments.csv", index=False)

print("✅ Comments successfully fetched and saved to data/youtube_comments.csv")
