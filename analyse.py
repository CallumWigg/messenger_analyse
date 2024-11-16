import collections
import json
import os
import random
import re
import sys
import time
from datetime import datetime
from io import BytesIO
import unicodedata  
import numpy as np

import emoji
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import networkx as nx
import requests
from PIL import Image
from wordcloud import WordCloud, STOPWORDS

# Consistent color palette
PALETTE = "viridis"
PERSON_COLORS = {}

def create_figure(figsize=(12, 9), dpi=400):
    """Creates and returns a matplotlib figure with specified settings."""
    fig = plt.figure(figsize=figsize, dpi=dpi)
    plt.tight_layout()
    plt.style.use("dark_background")
    return fig

def save_figure(filename, outdir):
    """Saves the current figure to the specified output directory."""
    plt.savefig(os.path.join(outdir, filename), bbox_inches='tight', dpi=400)
    plt.close()


def get_messages_and_participants(input_dir):
    """Loads messages and participants from JSON files in the input directory."""
    messages = []
    participants = set()

    for filename in os.listdir(input_dir):
        if filename.endswith(".json") and filename.startswith("message_"):
            file_path = os.path.join(input_dir, filename)
            with open(file_path, "rb") as file:
                raw = file.read().replace(b"\u00e2\u0080\u0099", b"'") # Fix encoding issue directly on read
                data = json.loads(raw.decode(encoding="utf-8"))
                messages.extend(data.get("messages", []))
                for participant in data["participants"]:
                    participants.add(participant["name"])

    return messages, participants

def count_messages(messages, key_func):
    """Generic function to count messages based on a key function."""
    data = collections.defaultdict(lambda: 0)
    for m in messages:
        key = key_func(m)
        if key:
            data[key] += 1
    return data

def count_words(messages, key_func):
    """Counts words in messages based on a key function."""
    data = collections.defaultdict(lambda: 0)
    for m in messages:
        key = key_func(m)
        if key:
            data[key] += len(m["content"].split())
    return data


def plot_pie_chart(data, title, outdir, threshold_percent=1):  # Added threshold
    labels = sorted(data.keys(), key=lambda k: data[k], reverse=True) # Sort labels for combining later
    values = [data[label] for label in labels]
    total_value = sum(values)
    
    # Combine small slices
    other_labels = []
    other_value = 0
    new_labels = []
    new_values = []
    
    for label, value in zip(labels, values):
        percent = (value / total_value) * 100
        if percent < threshold_percent:
            other_value += value
            other_labels.append(label)
        else:
            new_labels.append(label)
            new_values.append(value)

    if other_value > 0:
        new_labels.append(", ".join(other_labels))  # Join names with commas
        new_values.append(other_value)
    colors = [PERSON_COLORS.get(label, 'gray') for label in new_labels]
    *_, autotexts = plt.pie(new_values, labels=new_labels, autopct="", colors=colors) # Pass colors here

    for i, a in enumerate(autotexts):
        value = new_values[i]
        percentage = (value / sum(new_values)) * 100
        a.set_text(f"{value}\n({percentage:.1f}%)")  
    
    plt.title(title)
    save_figure(f"{title.lower().replace(' ', '_')}.png", outdir)

def time_series_data(messages, time_format="%b %y"):
    """Processes message timestamps and groups by person and time period."""
    people = set()
    data = collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    for m in messages:
        timestamp = m.get("timestamp_ms")
        name = m.get("sender_name")

        if not (timestamp and name):
            continue

        people.add(name)

        timestamp //= 1000
        timeblock = time.strftime(time_format, time.gmtime(timestamp))
        data[timeblock][name] += 1

    sorted_timeblocks = sorted(data.keys(), key=lambda k: time.mktime(time.strptime(k, time_format)))
    return list(people), data, sorted_timeblocks
    
def stackplot(messages, outdir):
    """Creates and saves a stackplot of message counts over time."""
    people, data, timeblocks = time_series_data(messages)
    ys = [[data[timeblock][name] for timeblock in timeblocks] for name in sorted(people)]

    create_figure()
    plt.stackplot(timeblocks, *ys, labels=sorted(people), antialiased=True, 
             colors=[PERSON_COLORS.get(p, 'gray') for p in sorted(people)]) 
    plt.legend(loc="upper left")
    plt.xticks(rotation=90)
    save_figure("stackplot.png", outdir)

def monthly_stacked_bar(messages, outdir):
    """Creates and saves a monthly stacked bar chart of message counts."""
    people, data, timeblocks = time_series_data(messages)

    create_figure()
    indexes = range(len(timeblocks))
    bottom = [0] * len(timeblocks)
    for name in sorted(people):
        person_data = [data[y][name] for y in timeblocks]
        plt.barh(indexes, person_data, left=bottom, label=name, 
         color=PERSON_COLORS.get(name, 'gray'))
        bottom = [bottom[i] + person_data[i] for i in indexes]

    plt.yticks(indexes, timeblocks)
    plt.legend()
    save_figure("monthly_stacked_bar.png", outdir)


def monthly_line(messages, outdir):
     """Creates and saves a line chart of monthly message counts per person."""
     people, data, timeblocks = time_series_data(messages)

     create_figure()
     for name in sorted(people):
         plt.plot(timeblocks, [data[x][name] for x in timeblocks], label=name, 
         color=PERSON_COLORS.get(name, 'gray'))
     plt.xticks(rotation=90)
     plt.ylabel("Messages Sent")
     plt.title("Messages by Month")
     plt.legend()
     save_figure("monthly_lines.png", outdir)


def words_per_message(messages, outdir):
    """Creates and saves a bar chart of words per message per person."""
    message_count = count_messages(messages, lambda m: m.get("sender_name"))
    words_count = count_words(messages, lambda m: m.get("sender_name"))

    words_per_message = {
        p: words_count[p] / message_count[p] if message_count[p] else 0 for p in message_count
    }

    create_figure()
    people = sorted(words_per_message.keys())
    values = [words_per_message[p] for p in people]
    plt.bar(people, values, color=[PERSON_COLORS.get(p, 'gray') for p in people]) 
    plt.xticks(rotation=45, ha='right')
    plt.xlabel("Sender")
    plt.ylabel("Words per Message")
    plt.title("Words per Message")
    save_figure("words_per_message.png", outdir)


def wordcloud(messages, outdir):
    """Generates and saves a word cloud image."""
    corpus = "\n".join([m["content"] for m in messages])
    corpus = re.sub(r'[^\x00-\x7F]+', '', corpus)  # Remove non-ASCII characters
    corpus = unicodedata.normalize('NFKD', corpus).encode('ascii', 'ignore').decode('ascii')  # Normalize Unicode 
    corpus = re.sub(r'[^a-zA-Z\s]', '', corpus)  # Remove remaining non-alphabetic characters (keep spaces)
    width = 3840   # Desired width
    height = 2160  # Desired height
    create_figure(figsize=(width/300, height/300), dpi=300)   
    wc = WordCloud(width=width, height=height, background_color="black").generate(corpus) 
    plt.imshow(wc, interpolation='bilinear')  # Use bilinear interpolation for smoother rendering
    plt.axis("off")

    plt.savefig(os.path.join(outdir, "wordcloud.png"), dpi=400, bbox_inches='tight')  # High DPI when saving
    plt.close()


def generate_nickname_wordcloud(nickname_data, output_path, palette):
    if not nickname_data:
        return

    # Combine consecutive identical nicknames and sum their durations
    combined_nicknames = {}
    current_nickname = None
    current_duration = 0
    for i in range(len(nickname_data)):
        nickname = nickname_data[i]["nickname"]
        duration = 0
        if i < len(nickname_data) -1 :
          duration = (nickname_data[i+1]["timestamp"] - nickname_data[i]["timestamp"]) // (1000 * 60 * 60 * 24)
        else:
          duration = (datetime.now() - datetime.fromtimestamp(nickname_data[i]["timestamp"] / 1000)).days

        if current_nickname is None:
            current_nickname = nickname
            current_duration = duration
        elif nickname == current_nickname:
            current_duration += duration
        else:
            combined_nicknames[current_nickname] = current_duration
            current_nickname = nickname
            current_duration = duration
    if current_nickname: #add the last one
      combined_nicknames[current_nickname] = current_duration

    # Generate wordcloud text while properly handling missing nicknames
    colors = plt.get_cmap(palette, max(len(combined_nicknames), 1)).colors
    color_map = {nickname: colors[i] for i, nickname in enumerate(combined_nicknames)}

    wordcloud = WordCloud(width=3000, height=3000,
                          background_color="black",
                          color_func=lambda *args, **kwargs: tuple(int(c * 255) for c in color_map.get(args[0], (0, 0, 0))),
                          stopwords=STOPWORDS,
                          min_font_size=20,
                          margin=0
                          ).generate_from_frequencies(combined_nicknames)

    plt.figure(figsize=(7, 7), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(output_path + ".png", dpi=400)
    plt.close()

def generate_and_save_wordclouds(nicknames, group_names, outdir, palette):
    """Generates and saves word clouds for nicknames and group names."""
    for person, entries in nicknames.items():
        entries.sort(key=lambda x: x["timestamp"])  # Ensure nicknames are sorted by time
        output_path = os.path.join(outdir, f"{person}_nicknames_wordcloud")
        generate_nickname_wordcloud(entries, output_path, palette)

    group_names.sort(key=lambda x: x["timestamp"])  # Ensure group names are sorted by time
    output_path = os.path.join(outdir, "group_names_wordcloud")
    generate_nickname_wordcloud(group_names, output_path, palette)


def markov_generate_message(data):
    """Generates a message using a Markov chain model."""
    message = ""
    word = "^"
    while word != "$":
        if word not in ".,":
            message += " "
        message += word
        word = random.choices(population=list(data[word]), weights=list(data[word].values()), k=1)[0]
    return message[3:].capitalize()


def markov_chain(messages, outdir):
    """Builds a Markov chain model and generates messages for each person."""
    data = collections.defaultdict(
        lambda: collections.defaultdict(lambda: collections.defaultdict(lambda: 0))
    )
    pattern = re.compile(r"[a-z\'\-]+|[.,]+")
    for m in messages:
        prev = "^"
        for word in re.findall(pattern, m["content"].lower()):
            data[m["sender_name"]][prev][word] += 1
            prev = word
        data[m["sender_name"]][prev]["$"] += 1

    generated = {}
    for name in data:
        generated[name] = [markov_generate_message(data[name]) for _ in range(10)]

    with open(os.path.join(outdir, "markov_messages.json"), "w") as f:
        json.dump(generated, f, indent=4)


def count_reactions(messages, key_func):
    """Generic function to count reactions based on a key function."""
    data = collections.defaultdict(lambda: 0)
    for m in messages:
        for reaction in m.get("reactions", []):
            key = key_func(m, reaction)
            if key:
                data[key] += 1
    return data



def count_reactions_by_person(messages):
    return count_reactions(messages, lambda _, r: r.get("actor"))

def count_reactions_received_by_person(messages):
    return count_reactions(messages, lambda m, _: m.get("sender_name"))

def pie_chart_reactions_given(messages, outdir):
    plot_pie_chart(count_reactions_by_person(messages), "Total Reactions Given", outdir)

def pie_chart_reactions_received(messages, outdir):
    plot_pie_chart(count_reactions_received_by_person(messages), "Total Reactions Received", outdir)


def garbage_message(m):
    """Checks if a message is a system-generated message or other non-user message"""
    text = m.get("content")
    if not text:
        return True

    chat_action_phrases = [
        " in the poll.",
        " created a poll: ",
        " responded with ",
        " created the reminder: ",
        " created the group.",
        " created a plan.",
        " set the nickname for ",
        " to your message "
    ]

    return any(phrase in text for phrase in chat_action_phrases)


def good_message(m):
    return not garbage_message(m)



def timestamp_to_date(timestamp):
    """Converts a timestamp to a date string."""
    return datetime.fromtimestamp(timestamp / 1000).strftime("%d/%m/%Y")

def save_nicknames(messages, participants, input_dir, outdir, nicknames):
    """Extracts and saves nickname history from messages."""

    for message in messages:
        content = message.get("content", "")
        timestamp = message.get("timestamp_ms", 0)
        sender_name = message.get("sender_name")

        if "set the nickname for" in content:
            parts = content.split(" set the nickname for ")
            nickname_parts = parts[1].split(" to ")
            person = nickname_parts[0]
            nickname = " ".join(nickname_parts[1:])
            
            nicknames[person] = nicknames.get(person, []) + [{"timestamp": timestamp, "nickname": nickname}]
            participants.discard(person) 

        elif "set your nickname to" in content or (
            "set the nickname for" in content and " to " in content and sender_name != "owner" 
        ):
            if "set your nickname to" in content:
                 parts = content.split("set your nickname to")
                 nickname = parts[1].strip(".") if len(parts) == 2 else ""

            else: 
                 nickname = content.split(" to ")[1].rstrip(".")
            
            nicknames["owner"] = nicknames.get("owner", []) + [{"timestamp": timestamp, "nickname": nickname}]  # Accumulate nicknames


    if len(participants) == 1:
        owner = participants.pop()
    else:                       
        owner = next((person for person, entries in nicknames.items() if any("set your nickname to" in entry.get("nickname", "") or 
                     ("set the nickname for" in entry.get("nickname", "") and entry.get("nickname", "").split()[2] == "owner") for entry in entries)), None)

        if not owner:       
            owner = next((message.get("sender_name") for message in messages if message.get("sender_name")), None)

    def format_nicknames(nickname_entries):
        """Helper function to format nickname entries with duration."""
        output = ""
        longest_standing = {"days": 0} # Store longest duration details


        for i in range(len(nickname_entries) - 1):
            current = nickname_entries[i]
            next_ = nickname_entries[i + 1]
            days_diff = (next_["timestamp"] - current["timestamp"]) // (1000 * 60 * 60 * 24)

            if days_diff > longest_standing["days"]:
                longest_standing["nickname"] = current["nickname"]
                longest_standing["days"] = days_diff
                longest_standing["start"] = timestamp_to_date(current["timestamp"])
                longest_standing["end"] = timestamp_to_date(next_["timestamp"])

        if nickname_entries: # Handle last nickname, potentially still active
            final_entry = nickname_entries[-1]
            final_duration = (datetime.now() - datetime.fromtimestamp(final_entry["timestamp"] / 1000)).days

            if final_duration > longest_standing["days"]:
                longest_standing["nickname"] = final_entry["nickname"]
                longest_standing["days"] = final_duration
                longest_standing["start"] = timestamp_to_date(final_entry["timestamp"])
                longest_standing["end"] = datetime.now().strftime("%d/%m/%Y")

        if longest_standing.get("nickname"):  # Output if a longest-standing nickname exists
            output += f"Longest Standing Nickname: {longest_standing['nickname']} " \
                      f"({longest_standing['days']} days, from {longest_standing['start']} " \
                      f"to {longest_standing['end']})\n"

        for entry in nickname_entries:
            output += f"{timestamp_to_date(entry['timestamp'])} - {entry['nickname'].rstrip('.')}\n"

        return output + "\n"

    output = ""

    if owner and owner in nicknames:  # Check if owner exists and has nicknames
        owner_entries = sorted(nicknames[owner], key=lambda x: x["timestamp"])
        output += f"{owner} ({len(owner_entries)} nicknames):\n"
        output += format_nicknames(owner_entries)


    for person, entries in nicknames.items():
        if person == owner:  # Skip the owner since they're already handled
            continue

        entries.sort(key=lambda x: x["timestamp"]) # Sort entries by timestamp
        output += f"{person} ({len(entries)} nicknames):\n"
        output += format_nicknames(entries)

    with open(os.path.join(outdir, f"{os.path.basename(input_dir)}_nicknames.txt"), "w", encoding="utf-8", errors="replace") as file:
        file.write(output)

def save_group_names(messages, input_dir, outdir, group_names):
    """Extracts and saves group name history from messages."""

    group_names = []

    for message in messages:
        content = message.get("content", "")
        timestamp = message.get("timestamp_ms", 0)
        sender_name = message.get("sender_name")

        match = re.search(r"(?:You|.*?) named the group (.*?)\.", content)
        if match:
            group_name = match.group(1).encode('latin1').decode('utf-8')  # Decode for emojis
            group_names.append({"timestamp": timestamp, "name": group_name, "sender": sender_name}) # Append to the list passed as an argument

    group_names.sort(key=lambda x: x["timestamp"])  # Sort by timestamp

    def format_group_names(group_name_entries):
        """Helper function to format group name entries with duration."""
        output = ""
        longest_standing = {"days": 0}

        for i in range(len(group_name_entries) - 1):
            current = group_name_entries[i]
            next_ = group_name_entries[i + 1]
            days_diff = (next_["timestamp"] - current["timestamp"]) // (1000 * 60 * 60 * 24)

            if days_diff > longest_standing["days"]:
                longest_standing["name"] = current["name"]
                longest_standing["sender"] = current.get("sender")
                longest_standing["days"] = days_diff
                longest_standing["start"] = timestamp_to_date(current["timestamp"])
                longest_standing["end"] = timestamp_to_date(next_["timestamp"])

        if group_name_entries:
            final_entry = group_name_entries[-1]
            final_duration = (datetime.now() - datetime.fromtimestamp(final_entry["timestamp"] / 1000)).days
            if final_duration > longest_standing["days"]:
                longest_standing["name"] = final_entry["name"]
                longest_standing["sender"] = final_entry.get("sender")
                longest_standing["days"] = final_duration
                longest_standing["start"] = timestamp_to_date(final_entry["timestamp"])
                longest_standing["end"] = datetime.now().strftime("%d/%m/%Y")


        if longest_standing.get("name"):
            output += f"Longest Standing Name: {longest_standing['name']} " \
                      f"(by {longest_standing['sender']}, " \
                      f"{longest_standing['days']} days, from {longest_standing['start']} " \
                      f"to {longest_standing['end']})\n\n"

        for entry in group_name_entries:
            output += f"{timestamp_to_date(entry['timestamp'])} - {entry['name']} (by {entry['sender']})\n"

        return output

    output = f"Group Names ({len(group_names)} total):\n\n"
    output += format_group_names(group_names)

    # Save the output to a file
    with open(os.path.join(outdir, f"{os.path.basename(input_dir)}_group_names.txt"), "w", encoding="utf-8", errors="replace") as file:
        file.write(output)


def plot_top_reactions(messages, outdir, title, reaction_func):
    """Generic function to plot top reactions (given or received)."""
    reaction_data_raw = reaction_func(messages)
    # Convert to dict of dicts if necessary:
    reaction_data = {}
    for person, data in reaction_data_raw.items():
        if isinstance(data, dict):
            reaction_data[person] = data
        elif isinstance(data, tuple):
            reaction_data[person] = {data[1]: data[0]}
        else:
            reaction_data[person] = {'default': data} # Handle case where count is not in a tuple for consistency

    total_reactions = {person: sum(reactions.values()) for person, reactions in reaction_data.items()}
    all_reactions = collections.defaultdict(int)
    for person_reactions in reaction_data.values():
        for emoji, count in person_reactions.items():
            all_reactions[emoji] += count
    top_10_overall = sorted(all_reactions.items(), key=lambda x: x[1], reverse=True)[:10]

    top_reactions = {}
    for person, reactions in reaction_data.items():
        top_3 = sorted(reactions.items(), key=lambda x: x[1], reverse=True)[:3]
        if top_3:
            top_reactions[person] = top_3

    top_reactions = dict(sorted(top_reactions.items(), key=lambda x: total_reactions[x[0]], reverse=True))

    if not top_reactions:
        return

    # Set emoji-supporting font 
    plt.rcParams['svg.fonttype'] = 'none'
    for font in ['Segoe UI Emoji', 'Apple Color Emoji', 'Noto Color Emoji']:
        try:
            plt.rcParams['font.family'] = font
            break
        except:
            continue


    create_figure(figsize=(16, 8))
    num_people = len(top_reactions)
    group_width = 0.8
    bar_width = group_width / 3
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']

    for person_idx, (person, top_3) in enumerate(top_reactions.items()):
        for reaction_idx, (reaction_emoji, count) in enumerate(top_3):
            x_pos = person_idx + (reaction_idx - 1) * bar_width
            plt.bar(x_pos, count, bar_width, color=colors[reaction_idx], alpha=0.8)
            plt.text(x_pos, count + (plt.ylim()[1] - plt.ylim()[0]) * 0.02, f"{reaction_emoji}\n{count}", ha='center', va='bottom', fontsize=16)

    plt.xticks(range(num_people), top_reactions.keys(), rotation=45, ha='right', fontsize=10)
    plt.ylabel('Number of Reactions', fontsize=12)
    plt.title(title, fontsize=14, pad=20)

    legend_text = [f'{emoji} {count:,}' for emoji, count in top_10_overall]
    plt.legend(legend_text, title='Top 10 Overall Reactions', title_fontsize=12, fontsize=12, loc='center left', bbox_to_anchor=(1.02, 0.5), borderaxespad=0)
    save_figure(f"{title.lower().replace(' ', '_')}.png", outdir) # Save as png and svg
    plt.savefig(os.path.join(outdir, f"{title.lower().replace(' ', '_')}.svg"), bbox_inches='tight')

def count_reactions_by_type_per_person(messages):
    data = collections.defaultdict(dict)  # Use dict directly
    for m in messages:
        for reaction in m.get("reactions", []):
            actor = reaction.get("actor")
            reaction_emoji = reaction.get("reaction", "").encode('latin1').decode('utf-8')
            if actor and reaction_emoji:
                if actor not in data:  # Initialize inner dictionary if needed
                    data[actor] = {}
                data[actor][reaction_emoji] = data[actor].get(reaction_emoji, 0) + 1 # Count reactions
    return data  # Always return dict of dicts

def count_reactions_received_by_type_per_person(messages):
    data = collections.defaultdict(dict)
    for m in messages:
        sender = m.get("sender_name")
        for reaction in m.get("reactions", []):
            reaction_emoji = reaction.get("reaction", "").encode('latin1').decode('utf-8')
            if sender and reaction_emoji:
                if sender not in data:  # Initialize inner dict
                    data[sender] = {}
                data[sender][reaction_emoji] = data[sender].get(reaction_emoji, 0) + 1 
    return data

def plot_top_reactions_by_person(messages, outdir):
    plot_top_reactions(messages, outdir, 'Top 3 Reactions SENT', 
                      lambda msgs: dict(count_reactions_by_type_per_person(msgs)))

def plot_top_reactions_received_by_person(messages, outdir):
    plot_top_reactions(messages, outdir, 'Top 3 Reactions RECEIVED',
                      lambda msgs: dict(count_reactions_received_by_type_per_person(msgs)))


def plot_ratio_chart(messages, outdir, title, count1_func, count2_func, ylabel, threshold=None, threshold_label=None, target_line=None):
    """Generic function to plot reaction/message ratios."""
    count1 = count1_func(messages)  # Renamed to given
    count2 = count2_func(messages)  # Renamed to received
    total_count1 = sum(count1.values())

    ratio_data = {}
    for person in set(count1.keys()) | set(count2.keys()):
        given = count1.get(person, 0)  # Use given instead of c1
        if threshold is not None and given < threshold * total_count1:
            continue

        received = count2.get(person, 0)  # Use received instead of c2
        if given == 0:
            given = 0.1  # Avoid division by zero

        ratio = received / given
        ratio_data[person] = {'ratio': ratio, 'given': given, 'received': received}  # Use received and given

    sorted_data = dict(sorted(ratio_data.items(), key=lambda x: x[1]['ratio'], reverse=True))

    create_figure(figsize=(12, 6))
    people = list(sorted_data.keys())
    ratios = [data['ratio'] for data in sorted_data.values()]
    bars = plt.bar(people, ratios, color=[PERSON_COLORS.get(p, 'gray') for p in people])

    plt.xticks(rotation=45, ha='right')
    plt.ylabel(ylabel)
    plt.title(title)

    for bar, person in zip(bars, people):
        data = sorted_data[person]
        stats_text = (f'Ratio: {data["ratio"]:.2f}\n'
            f'Received: {int(data["received"])}\n'  # Changed count2 to received
            f'Given: {int(data["given"] if data["given"] != 0.1 else 0)}')  # Changed count1 to given

        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), stats_text, ha='center', va='bottom')

    if target_line is not None:
        plt.axhline(y=target_line, color='gray', linestyle='--', alpha=0.5)
    plt.text(-0.5, target_line or (threshold if threshold is not None else 1.0), threshold_label or 'Equal Give/Receive', va='bottom', alpha=0.5)

    save_figure(f"{title.lower().replace(' ', '_')}.png", outdir)


def plot_reaction_ratio(messages, outdir):
    plot_ratio_chart(
        messages, 
        outdir, 
        "Reactions Received vs Reactions Sent",
        count_reactions_by_person, 
        count_reactions_received_by_person,
        'Reactions Received / Reactions Given',
        threshold=0.02,  # Threshold for reactions/reactions is now 0.02
        threshold_label='Equal Give/Receive',
        target_line=1.0  # Line at 1.0 for reactions/reactions
    )


def plot_message_reaction_ratio(messages, outdir):
    message_counts = count_messages(messages, lambda m: m.get("sender_name"))
    reactions_received = count_reactions_received_by_person(messages)
    total_messages = sum(message_counts.values())
    message_threshold = 0.02 * total_messages
    ratio_data = {}
    for person in set(message_counts.keys()) | set(reactions_received.keys()):
        sent = message_counts.get(person, 0)
        if sent < message_threshold:
            continue
        received = reactions_received.get(person, 0)
        if sent == 0:
            sent = 0.1 
        ratio = received / sent
        ratio_data[person] = {'ratio': ratio, 'sent': sent, 'received': received}
    # Calculate median ratio (outside plot_ratio_chart)
    ratios = [data['ratio'] for data in ratio_data.values()]
    median_ratio = np.median(ratios) if ratios else 0 # Handle empty list

    plot_ratio_chart(
        messages,
        outdir,
        "Average Reactions Received per Message Sent",
        lambda msgs: count_messages(msgs, lambda m: m.get("sender_name")),
        count_reactions_received_by_person,
        'Reactions Received / Messages Sent',
        threshold=0.02,
        threshold_label="Median Reactions/Messages",  # Update label
        target_line=median_ratio    # Pass median ratio to plot_ratio_chart
    )


def favourite_person_reactions(messages, outdir):
    """Visualizes each person's favourite person based on reaction counts."""
    reaction_counts = collections.defaultdict(lambda: collections.defaultdict(int))
    for m in messages:
        sender = m.get("sender_name")
        for reaction in m.get("reactions", []):
            reactor = reaction.get("actor")
            if reactor and sender:
                reaction_counts[reactor][sender] += 1


    favourites = {reactor: max(reacted_to, key=reacted_to.get) # Check if reacted_to is not empty before getting max
                    for reactor, reacted_to in reaction_counts.items() if reacted_to}


    graph = nx.DiGraph()
    graph.add_edges_from([(reactor, favourite) for reactor, favourite in favourites.items()])

    num_participants = len(graph.nodes)
    fig_size = (max(8, num_participants * 0.5), max(6, num_participants * 0.4))
    create_figure(figsize=fig_size)

    pos = nx.spring_layout(graph, k=2.5 / (num_participants**0.5) if num_participants > 0 else 1, seed=42) # Avoid division by zero
    colors = plt.get_cmap('hsv', num_participants) if num_participants > 0 else plt.get_cmap('hsv', 1)  # Handle case where no participants have reacted

    nx.draw_networkx_nodes(graph, pos, node_size=700,
                           node_color=[colors(i) for i in range(num_participants)],
                           alpha=0.8, edgecolors='black', linewidths=1)

    nx.draw_networkx_labels(graph, pos, font_size=12, font_color="white", font_family="sans-serif")
    nx.draw_networkx_edges(graph, pos, arrows=True, arrowstyle="-|>", arrowsize=20, edge_color="gray", width=1.5)

    legend_text = [f"{reactor}'s favourite person is {favourite}" for reactor, favourite in favourites.items()]
    legend_handles = [plt.Line2D([0], [0], marker=f"${i}$", color='w', label=text, 
                                   markerfacecolor=colors(i), markersize=10)
                      for i, text in enumerate(legend_text)]
    plt.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)

    plt.title("Favourite Person by Reactions Received", fontsize=16, color="white", loc='center')
    plt.axis("off")
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    save_figure("favourite_person.png", outdir)




def most_reacted_messages(messages, outdir):
    """Creates an image of messages with most reactions (excluding self reactions)."""

    message_reactions = {}
    for m in messages:
        message_id = m.get("timestamp_ms")
        sender = m.get("sender_name")
        reactions = m.get("reactions", [])

        if not message_id:
            continue

        react_count = sum(1 for reaction in reactions if reaction.get("actor") and reaction.get("actor") != sender)
        message_reactions[message_id] = {"content": m.get("content"), "react_count": react_count}


    if not message_reactions:
        return

    reaction_counts = sorted(list(set(entry["react_count"] for entry in message_reactions.values())), reverse=True)
    top_reaction_counts = reaction_counts[:2]

    if not top_reaction_counts or top_reaction_counts[0] == 0:
        return


    most_reacted = []
    for react_count in top_reaction_counts:
        messages_with_count = [entry["content"] for entry in message_reactions.values() if entry["react_count"] == react_count]
        most_reacted.extend([(react_count, msg) for msg in messages_with_count])

    if most_reacted:
        from PIL import Image, ImageDraw, ImageFont

        viridis = plt.get_cmap('viridis')

        img_height = sum(50 for _, _ in most_reacted) + 50 + 50 * len(top_reaction_counts)
        img = Image.new('RGB', (800, img_height), color = 'black')
        d = ImageDraw.Draw(img)


        try:
            font = ImageFont.truetype("seguiemj.ttf", 16)
        except:
            font = ImageFont.truetype("arial.ttf", 16)
        
        title_font = ImageFont.truetype("arial.ttf", 20)

        y = 25
        start_index = 0

        for react_count in top_reaction_counts:
            color = viridis(react_count / max(top_reaction_counts))
            title_color = tuple(int(c * 255) for c in color[:3])
            d.text((25, y), f"Messages with {react_count} reactions:", font=title_font, fill=title_color)
            y += 50

            end_index = start_index + len([count for count, _ in most_reacted if count == react_count])

            for i in range(start_index, end_index):
                message = most_reacted[i][1]
                try:
                    d.text((25, y), message, font=font, fill="white") # consistent message color
                except:
                    pass
                y += 50
            start_index = end_index

        img.save(os.path.join(outdir, "most_reacted_messages.png"), dpi=(400, 400))

def process_and_save_names(all_messages, participants, input_dir, outdir, palette):
    """Processes, saves, and generates wordclouds for nicknames and group names."""

    nicknames = {}
    group_names = []

    save_nicknames(all_messages, participants.copy(), input_dir, outdir, nicknames)  # Pass nicknames by reference
    save_group_names(all_messages, input_dir, outdir, group_names)  # Pass group_names by reference
    
    generate_and_save_wordclouds(nicknames, group_names, outdir, palette)


def main():
    input_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(__file__)

    all_messages, participants = get_messages_and_participants(input_dir)
    good_messages = list(filter(good_message, all_messages))

    global PERSON_COLORS
    outdir = "out"
    os.makedirs(outdir, exist_ok=True)

    num_participants = len(participants)
    colors = plt.get_cmap(PALETTE, num_participants).colors
    PERSON_COLORS = dict(zip(participants, colors))

    process_and_save_names(all_messages, participants, input_dir, outdir, PALETTE)

    outputs = [
        monthly_line,
        (pie_chart, good_messages),
        (pie_chart_words, good_messages),
        words_per_message,
        stackplot,
        markov_chain,
        pie_chart_reactions_given,
        pie_chart_reactions_received,
        plot_reaction_ratio,
        plot_message_reaction_ratio,
        plot_top_reactions_received_by_person,
        favourite_person_reactions,
        most_reacted_messages,
        (plot_top_reactions_by_person, good_messages),
        wordcloud,
    ]

    for item in outputs:
        create_figure()
        if isinstance(item, tuple):
            func, *args = item
            if func == pie_chart or func == pie_chart_words:
                func(*args, outdir)  # Correct call for pie charts
            else:
                func(*args, outdir)
        else:
            item(good_messages, outdir)

# Alias functions for consistency (messages, words)
pie_chart = lambda messages, outdir: plot_pie_chart(count_messages(messages, lambda m: m.get("sender_name")), "Total Messages Sent", outdir)
pie_chart_words = lambda messages, outdir: plot_pie_chart(count_words(messages, lambda m: m.get("sender_name")), "Total Words Sent", outdir)


if __name__ == "__main__":
    main()
