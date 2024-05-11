import logging

import etlUtils.etl_shared
from tqdm.auto import tqdm
import os
import json
import threading
from datetime import timedelta
#import srt
import requests
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound
import concurrent.futures
from urllib.parse import urlparse, parse_qs
import random
from typing import Dict, List, Set, Tuple
from collections import namedtuple

# A dict of videoId => Reason
temp_failed_videos = {}
temp_translated_videos = {}

thread_local = threading.local()

SubsExtractResults = namedtuple('SubsExtractResults', ['data', 'failed_videos', 'translated_videos'])


loggermain = logging.getLogger("main")


def add_video_to_local_db(documents_json):
    import json

    res = []
    for d in documents_json:
        doc_selected = {#'_id' : str(d['_id']),
        'id': d['id'],
        'type' : d['type'],
        'text' : d['text'],
        'title' : d['metadata']['title'],
        'source' : d['metadata']['source'],
        'chapter-title' : d['metadata']['chapter-title']}

        res.append(json.dumps(doc_selected)+ '\n')
    return res


translatable_languages = ['German',
'Spanish',
'Italian',
'Roma',
'Romanian',
'Russian',
'Dutch',
'Danish']


def is_translatable_language(lang_in: str) -> bool:
    anygood: bool = False
    for lang in translatable_languages:
        if lang in lang_in:
            anygood = True
            break
    return anygood


def clean_videos():
    print("Cleaning videos")
    failed_videos_path = os.environ["VIDEOS_LOCAL_JSON_DB_FAILED"]
    translated_videos_path = os.environ["VIDEOS_LOCAL_JSON_DB_TRANSLATED"]
    db_videos_path = os.environ["VIDEOS_LOCAL_JSON_DB"]
    db_videos_clean_path = os.environ["VIDEOS_LOCAL_JSON_DB_CLEANED"]

    # Collect all failed and translated movies ids
    all_ids_to_remove = set()
    with open(failed_videos_path, "r") as failed_videos_file:
        failed_videos_data = failed_videos_file.readlines()
        for failed_video in failed_videos_data:
            failed_video_data = json.loads(failed_video)
            all_ids_to_remove.add(failed_video_data["id"])

    with open(translated_videos_path, "r") as translated_videos_file:
        translated_videos_data = translated_videos_file.readlines()
        for translated_video in translated_videos_data:
            translated_video_data = json.loads(translated_video)
            all_ids_to_remove.add(translated_video_data["id"])

    # Clean the raw videos dataset
    with open(db_videos_clean_path, "w") as db_videos_clean_file:
        db_videos_clean_data = []
        with open(db_videos_path, "r") as db_videos_file:
            db_videos_data = db_videos_file.readlines()
            for db_video in tqdm(db_videos_data, total=len(db_videos_data), desc="Cleaning videos dataset", nrows=1):
                db_video_data = json.loads(db_video)
                if db_video_data["id"] not in all_ids_to_remove:
                    db_videos_clean_data.append(json.dumps(db_video_data) + "\n")
        db_videos_clean_file.writelines(db_videos_clean_data)


all_subtitles = []

#@stub.local_entrypoint()
def main(json_path="data/videos.json", collection=None, db=None, demoMode=False):
    global all_subtitles
    global temp_failed_videos
    temp_failed_videos = {}

    f = open(os.environ["VIDEOS_LOCAL_JSON_DB"], 'r+')
    f.truncate(0)
    f.close()

    global temp_translated_videos
    temp_translated_videos = {}

    with open(json_path) as f:
        video_infos = json.load(f)

    # Subsample in demo mode
    if demoMode:
        video_infos = etlUtils.etl_shared.subsampleForDemo(video_infos, etlUtils.etl_shared.DEMO_VIDEO_SAMPLES)

    # Extract subtitles and chapters
    # all_papers_texts = map(extract_pdf, paper_data)
    all_subtitles = []

    USE_PARALLEL = etlUtils.etl_shared.PARALLEL_USE

    def add_subtitle(myvideo_subs):
        global all_subtitles
        if myvideo_subs is None or len(myvideo_subs) == 0:
            return
        all_subtitles.append(myvideo_subs)

    # Write currently collected subtitles to the datastore
    def write_chunks_to_database(ending_index: int, chunk_index: int):
        global all_subtitles
        if len(all_subtitles) > 0:
            loggermain.debug(f"### Adding subtitles to database ending index: {ending_index} and chunk index: {chunk_index}")
        else:
            loggermain.debug(f"### NO subtitles to database ending index: {ending_index} and chunk index: {chunk_index}")
        documents = etlUtils.etl_shared.unchunk(all_subtitles)

        num_chunks = 10
        chunked_documents = etlUtils.etl_shared.chunk_into(documents, num_chunks)
        res = []
        jsonlines = []
        for myvideodoc in tqdm(chunked_documents, total=(num_chunks),
                               desc="Sending chuncks of videos to mongdb and local json"):
            r = etlUtils.etl_shared.add_to_mongo_db(myvideodoc)
            res.append(r)

            res_json = add_video_to_local_db(myvideodoc)
            jsonlines.append(res_json)

        jsonlines = etlUtils.etl_shared.unchunk(jsonlines)

        with open(os.environ["VIDEOS_LOCAL_JSON_DB"], 'a') as the_file:
            the_file.writelines(jsonlines)

        all_subtitles = []

    if USE_PARALLEL:
        from multiprocessing import Pool, TimeoutError

        n_processes = etlUtils.etl_shared.PARALLEL_NUM_PROCESSES
        with Pool(processes=n_processes) as pool:
            results: SubsExtractResults = etlUtils.etl_shared.imap_unordered_bar(extract_subtitles, video_infos, n_processes=n_processes, desc="Parallel extraction of subtitles: ")

        temp_failed_videos = {}
        temp_translated_videos = {}
        for res in tqdm(results, desc="Adding all subtitles gathered: "):
            if res == None or res == []:
                continue

            assert type(res) == SubsExtractResults, "Incorrect type of data "
            add_subtitle(res.data)
            temp_failed_videos.update(res.failed_videos)
            temp_translated_videos.update(res.translated_videos)


    else:
        # Continuous writing in small chuncks
        #START_DEBUG_INDEX = 5750
        num_videos = len(video_infos)
        num_chunks = 500
        chunk_counter = 0
        chunk_size = num_videos // num_chunks
        for videoidx, myvideo in enumerate(tqdm(video_infos, total=num_videos, desc="Sequential extract subtitles", leave=True, position=0)):
            #if videoidx < START_DEBUG_INDEX:
            #    continue

            # print("Extracting from video {}".format(myvideo))
            results: SubsExtractResults = extract_subtitles(myvideo) # temp_failed_videos and translated already filled...
            if results is None or results == []:
                continue
            add_subtitle(results.data)

            if videoidx > 0 and videoidx % chunk_size == 0 or videoidx == num_videos - 1:
                write_chunks_to_database(ending_index=videoidx, chunk_index=chunk_counter)
                chunk_counter +=1


    # Export a json with all failed ids and reasons
    with open(os.environ["VIDEOS_LOCAL_JSON_DB_FAILED"], 'w') as the_file:
        for failedKey, failedDetails in temp_failed_videos.items():
            jsonlinetoWrite = json.dumps({"id":failedKey, "reason": failedDetails})
            the_file.write(jsonlinetoWrite + "\n")

    with open(os.environ["VIDEOS_LOCAL_JSON_DB_TRANSLATED"], 'w') as the_translated_file:
        for transKey, transDetails in temp_translated_videos.items():
            jsonlinetoWrite = json.dumps({"id": transKey, "reason": transDetails})
            the_translated_file.write(jsonlinetoWrite + "\n")

    clean_videos()


def extract_subtitles(video_info) -> SubsExtractResults:
    if 'id' not in video_info:
        url_data = urlparse(video_info['url'])
        query = parse_qs(url_data.query)
        video_info['id'] = query["v"][0]

    video_id, video_title = video_info["id"], video_info["title"]
    subtitles = None
    try:
        subtitles = get_transcript(video_id)
    except  Exception as e:
        vid: str = video_id
        temp_failed_videos[vid] = str(e)
        lenght_info = len(temp_failed_videos[vid])
        temp_failed_videos[vid] = temp_failed_videos[vid][:min(lenght_info, 500)]
        # print(e)
        if type(e) == NoTranscriptFound:
            # Try for a translation
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            try:
                transcript = transcript_list.find_manually_created_transcript(['en'])
                assert transcript is None, "Could not find, but transcript is available!"
            except Exception as e:
                pass

            # YouTubeTranscriptApi.get_transcript(video_id)
            # iterate over all available transcripts
            for idx, transcript in enumerate(transcript_list):
                if not is_translatable_language(transcript.language):
                    add_suffix = f"original lang: {transcript.language} "
                    temp_failed_videos[vid] = add_suffix + temp_failed_videos[vid]
                    continue
                else:
                    subtitles = transcript.translate('en').fetch()
                    #if a != None and a != []:
                    #    subtitles = ' '.join([item_tr['text'] for item_tr in a])

            valueOut = temp_failed_videos.pop(vid)
            temp_translated_videos[vid] = valueOut
    if subtitles is None:
        return []

    chapters = get_chapters(video_id)
    if chapters is None:
        return []

    chapters = add_transcript(chapters, subtitles)

    if subtitles is not None:
        documents = create_documents(chapters, video_id, video_title)

    return SubsExtractResults(documents,
                              temp_failed_videos,
                              temp_translated_videos)


def get_transcript(video_id):
    return YouTubeTranscriptApi.get_transcript(video_id)


def get_chapters(video_id):
    base_url = "https://yt.lemnoslife.com"
    request_path = "/videos"

    params = {"id": video_id, "part": "chapters"} #, "timeout" : "10"}
    #"https://yt.lemnoslife.com/videos?part=chapters&id=NNgYId7b4j0"

    try:
        fullUrl = base_url + request_path + f"?part={params['part']}&id={params['id']}"
        response = requests.get(fullUrl, timeout=25) #requests.get(base_url + request_path, params=params)
        response.raise_for_status()
    except Exception as e:
        print(e)
        #workingfullUrl = "https://yt.lemnoslife.com/videos?part=chapters&id=E5bSumTAHZE"
        #response = requests.get(workingfullUrl)
        return None

    chapters = response.json()["items"][0]["chapters"]["chapters"]

    # "Video has no chapters" - add one single
    if len(chapters) == 0:
        chapters.append({"thumbnails":None, "time":0, "title":"entire clip"})


    for chapter in chapters:
        if chapter["thumbnails"] is not None:
            del chapter["thumbnails"]

    return chapters


def add_transcript(chapters, subtitles):
    for ii, chapter in enumerate(chapters):
        next_chapter = chapters[ii + 1] if ii < len(chapters) - 1 else {"time": 1e10}

        try:
            text = " ".join(
                [
                    seg["text"].strip()
                    for seg in subtitles
                    if seg["start"] >= chapter["time"]
                    and seg["start"] < next_chapter["time"]
                ])
        except Exception as e:
            print(f"Exception {e}. subtitles: {subtitles}, chapter: {chapter}")


        chapter["text"] = text

    return chapters


def create_documents(chapters, id, video_title):
    base_url = f"https://www.youtube.com/watch?v={id}"
    query_params_format = "&t={start}s"
    documents = []

    for chapter in chapters:
        text = chapter["text"].strip()
        start = chapter["time"]
        url = base_url + query_params_format.format(start=start)

        document = {"id": id,
                    "text": text,
                    "type": "videotranscript",
                    "metadata": {"source": url}
                    }

        document["metadata"]["title"] = video_title
        document["metadata"]["chapter-title"] = chapter["title"]
        document["metadata"]["full-title"] = f"{video_title} - {chapter['title']}"

        documents.append(document)

    documents = etlUtils.etl_shared.add_metadata(documents)

    return documents


def merge(subtitles, idx):

    new_content = combine_content(subtitles)

    # preserve start as timedelta
    new_start = seconds_float_to_timedelta(subtitles[0]["start"])
    # merge durations as timedelta
    new_duration = seconds_float_to_timedelta(sum(sub["duration"] for sub in subtitles))

    # combine
    new_end = new_start + new_duration

    return srt.Subtitle(index=idx, start=new_start, end=new_end, content=new_content)


def timestamp_from_timedelta(td):
    return int(td.total_seconds())


def combine_content(subtitles):
    contents = [subtitle["text"].strip() for subtitle in subtitles]
    return " ".join(contents) + "\n\n"


def get_charcount(subtitle):
    return len(subtitle["text"])


def seconds_float_to_timedelta(x_seconds):

    return timedelta(seconds=x_seconds)

if __name__ == "__main__":
    main()