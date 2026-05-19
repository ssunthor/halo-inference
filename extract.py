# Author: Scott Chrestman
# Copyright: 2025 GTT Communications, Inc.

import requests
import pandas as pd
import io
import getpass
import urllib3
from datetime import datetime, timedelta, timezone
import curses
import csv
import threading
import time
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import joblib
import os
import onnx
from onnxconverter_common import float16
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import StringTensorType, FloatTensorType
from sklearn.preprocessing import LabelEncoder
import argparse
import re
import numpy as np
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed, wait

# Import cuML if available for GPU-accelerated training
try:
    import cuml
    CUML_AVAILABLE = True
except ImportError:
    CUML_AVAILABLE = False

# Global lock for thread-safe UI updates
UI_LOCK = threading.Lock()

# --- Configuration ---
GRAYLOG_URL = "https://segraylog-eu.gt-t.net"
FIELDS = "timestamp,source,message"
DATA_FILE = "graylog_training_data.csv"
MODEL_DIR = "models"
LOG_FILE = "graylog_extraction_training.log"

# Model and encoder paths
VECTORIZER_PATH = os.path.join(MODEL_DIR, "vectorizer.joblib")
MODEL_PATH = os.path.join(MODEL_DIR, "model.joblib")
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "label_encoder.joblib")
IS_SECURITY_EVENT_VECTORIZER_PATH = os.path.join(MODEL_DIR, "is_security_event_vectorizer.joblib")
IS_SECURITY_EVENT_MODEL_PATH = os.path.join(MODEL_DIR, "is_security_event_model.joblib")
IS_SECURITY_EVENT_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "is_security_event_label_encoder.joblib")
EVENT_TYPE_VECTORIZER_PATH = os.path.join(MODEL_DIR, "event_type_vectorizer.joblib")
EVENT_TYPE_MODEL_PATH = os.path.join(MODEL_DIR, "event_type_model.joblib")
EVENT_TYPE_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "event_type_label_encoder.joblib")
PRIORITY_VECTORIZER_PATH = os.path.join(MODEL_DIR, "priority_vectorizer.joblib")
PRIORITY_MODEL_PATH = os.path.join(MODEL_DIR, "priority_model.joblib")
PRIORITY_LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, "priority_label_encoder.joblib")

# Graylog queries for data extraction
QUERIES = {
    "SECURITY_AUTHENTICATION_FAILURE": {
        "label": "SECURITY_AUTHENTICATION_FAILURE", "is_security_event": True, "event_type": "Authentication", "priority": "high",
        "queries": [
            'source:* AND program:sshd AND message:"Invalid user"',
            'source:* AND (EventID:4625) OR (message:/.*Failed password for.*/ OR message:/.*Invalid user.*/)',
        ]
    },
    "SECURITY_AUTHENTICATION_SUCCESS": {
        "label": "SECURITY_AUTHENTICATION_SUCCESS", "is_security_event": True, "event_type": "Authentication", "priority": "low",
        "queries": [
            'source:* AND program:sshd AND message:"session opened for user"',
            'source:* AND EventID:4624',
            'source:* AND program:sshd AND message:"Accepted publickey"',
        ]
    },
    "SECURITY_AUTHORIZATION_FAILURE": {
        "label": "SECURITY_AUTHORIZATION_FAILURE", "is_security_event": True, "event_type": "Authorization", "priority": "medium",
        "queries": [
            'source:* AND program:sudo AND message:"user NOT in sudoers"',
            'source:* AND program:mariadbd AND message:/.*denied.*/',
        ]
    },
    "SECURITY_FIREWALL_INTRUSION": {
        "label": "FIREWALL_INTRUSION", "is_security_event": True, "event_type": "Intrusion", "priority": "high",
        "queries": ['source:* AND message:"%FTD" AND message:"Intrusion Block"']
    },
    "SECURITY_FIREWALL_BLOCK": {
        "label": "FIREWALL_BLOCK", "is_security_event": True, "event_type": "Block", "priority": "medium",
        "queries": ['source:* AND message:"%FTD" AND message:"AccessControlRuleAction: Block"']
    },
    "SECURITY_DNS_FAILURE": {
        "label": "FIREWALL_DNS_FAIL", "is_security_event": False, "event_type": "DNS Failure", "priority": "low",
        "queries": ['source:* AND message:"%FTD" AND message:"DNS Lookup"']
    },
    "SECURITY_NORMAL": {
        "label": "SECURITY_NORMAL", "is_security_event": True, "event_type": "Normal Security Event", "priority": "low",
        "queries": [
            'source:* AND program:sudo AND NOT message:"user NOT in sudoers"'
        ]
    },
    "NON_SECURITY_NORMAL": {
        "label": "NON_SECURITY_NORMAL", "is_security_event": False, "event_type": "Normal Nonsecurity Event", "priority": "none",
        "queries": [
            'source:* AND (application_name:CRON OR application_name:systemd)',
            'source:* AND message:"HTTP/1.1" AND message:" 200 "',
            'source:* AND program:named AND message:"query (cache)"'
        ]
    },
    "EMAIL_SUCCESS": {
        "label": "EMAIL_SUCCESS", "is_security_event": False, "event_type": "Success Email Delivery", "priority": "none",
        "queries": [
            'source:* AND message:"gtt-smtp" AND message:"Queued mail for delivery"'
        ]
    },
    "EMAIL_FAILURE": {
        "label": "EMAIL_FAILURE", "is_security_event": False, "event_type": "Email Delivery Failure", "priority": "none",
        "queries": [
            'source:* AND message:"gtt-smtp" AND message:"5.4.1 Recipient address rejected"'
        ]
    }
}

def parse_time_delta(time_str):
    """Parses a time string (e.g., '1d', '2w', '3h') into a timedelta object."""
    match = re.match(r"(\d+)([hdwms])", time_str.lower())
    if not match:
        raise ValueError(f"Invalid time format: '{time_str}'. Use format like 1d, 3h, 2w, 30m.")
    
    value, unit = match.groups()
    value = int(value)
    
    unit_map = {'h': 'hours', 'd': 'days', 'w': 'weeks', 'm': 'minutes', 's': 'seconds'}
    return timedelta(**{unit_map[unit]: value})

class LogFetcher(threading.Thread):
    def __init__(self, query, category_name, label_value, is_security_event, event_type, priority, message_window, stats_dict, lock, timing_stats, timing_lock, time_from, time_to, limit, effective_total_rows, effective_total_lock):
        super().__init__()
        self.query = query
        self.category_name = category_name
        self.label_value = label_value
        self.is_security_event = is_security_event
        self.event_type = event_type
        self.priority = priority
        self.message_window = message_window
        self.stats_dict = stats_dict
        self.lock = lock
        self.timing_stats = timing_stats
        self.timing_lock = timing_lock
        self.time_from = time_from
        self.time_to = time_to
        self.limit = limit
        self.effective_total_rows = effective_total_rows
        self.effective_total_lock = effective_total_lock
        self.result_df = pd.DataFrame()
        self.error = None
        self.status = "running"
        self.max_y, self.max_x = message_window.getmaxyx()

    def _add_message_to_window(self, msg, color_pair):
        add_scrollable_message(self.message_window, msg, color_pair)

    def _make_graylog_request(self, url, params, headers=None, timeout=60, max_retries=1, delay=5):
        for attempt in range(max_retries):
            try:
                response = requests.get(
                    url, auth=(USERNAME, PASSWORD), params=params, headers=headers, verify=False, timeout=timeout
                )
                response.raise_for_status()
                return response
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    continue
                else:
                    raise e
        return None

    def run(self):
        # Phase 1: Get count
        count_url = f"{GRAYLOG_URL}/api/search/universal/absolute"
        count_params = {
            "query": self.query,
            "from": self.time_from,
            "to": self.time_to,
            "limit": 1,
            "fields": "message"
        }
        headers = {"Accept": "application/json"}
        
        retries = 0
        max_retries = 5

        while True:
            try:
                count_response = self._make_graylog_request(count_url, count_params, headers=headers, timeout=60)
                count_data = count_response.json()
                total_results = count_data.get("total_results", 0)

                if total_results > 0:
                    with self.lock:
                        self.stats_dict[self.category_name]['total'] += total_results
                    
                    effective_rows = total_results if self.limit is None else min(total_results, self.limit)
                    with self.effective_total_lock:
                        self.effective_total_rows[0] += effective_rows
                    break 
                else:
                    category_has_data = False
                    with self.lock:
                        if self.stats_dict[self.category_name]['total'] > 0:
                            category_has_data = True

                    if category_has_data:
                        if retries >= max_retries:
                            self._add_message_to_window(f"Query: {self.query}", curses.color_pair(2))
                            self._add_message_to_window(f"  -> No results after {retries} retries. Skipping (Category has data).", curses.color_pair(3))
                            self.status = "completed"
                            return

                        retries += 1
                        self._add_message_to_window(f"Query: {self.query}", curses.color_pair(2))
                        self._add_message_to_window(f"  -> No results. Retrying ({retries}/{max_retries}) in 10s...", curses.color_pair(3))
                    else:
                        self._add_message_to_window(f"Query: {self.query}", curses.color_pair(2))
                        self._add_message_to_window("  -> No results (Category empty). Retrying in 10s...", curses.color_pair(3))
                    
                    time.sleep(10)
                    continue

            except Exception as e:
                self._add_message_to_window(f"Query: {self.query}", curses.color_pair(2))
                self._add_message_to_window(f"  -> Error: {e}. Retrying in 10s...", curses.color_pair(4))
                time.sleep(10)
                continue

        # Phase 2: Get data
        try:
            self._add_message_to_window(f"Query: {self.query}", curses.color_pair(2))

            export_url = f"{GRAYLOG_URL}/api/search/universal/absolute/export"
            export_params = {
                "query": self.query,
                "from": self.time_from,
                "to": self.time_to,
                "fields": FIELDS,
            }
            if self.limit is not None:
                export_params['limit'] = self.limit
            
            start_time = time.time()
            response = self._make_graylog_request(export_url, export_params, timeout=300)
            end_time = time.time()

            duration = end_time - start_time

            if not response.headers.get('content-type', '').startswith('text/csv'):
                self._add_message_to_window(f"  -> Unexpected content type: {response.headers.get('content-type')}", curses.color_pair(4))
                self.status = "error"
                return

            df = pd.read_csv(io.StringIO(response.text))
            retrieved_count = len(df)

            with self.timing_lock:
                self.timing_stats.append((duration, retrieved_count))

            with self.lock:
                self.stats_dict[self.category_name]['retrieved'] += retrieved_count

            self._add_message_to_window(f"  -> Retrieved {retrieved_count} rows in {duration:.2f}s.", curses.color_pair(3))

            df["label"] = self.label_value
            df["is_security_event"] = self.is_security_event
            df["event_type"] = self.event_type
            df["priority"] = self.priority
            self.result_df = df
            self.status = "completed"

        except Exception as e:
            self._add_message_to_window(f"Query: {self.query}", curses.color_pair(2))
            add_scrollable_message(self.message_window, f"  -> Error fetching data: {e}", curses.color_pair(4))
            self.error = e
            self.status = "error"

def add_scrollable_message(message_window, msg, color_pair):
    with open(LOG_FILE, "a") as f:
        f.write(f"{datetime.now().isoformat()} - {msg}\n")

    with UI_LOCK:
        max_y, max_x = message_window.getmaxyx()
        current_y, current_x = message_window.getyx()
        if current_y >= max_y - 2:
            message_window.scroll(1)
            message_window.move(max_y - 2, 1)
        else:
            message_window.move(current_y + 1, 1)

        available_width = max_x - 2
        truncated_msg = msg
        if len(msg) > available_width:
            truncated_msg = msg[: available_width - 3] + "..."

        message_window.clrtoeol()
        message_window.addstr(truncated_msg, curses.color_pair(color_pair) | curses.A_BOLD)
        message_window.box()
        message_window.refresh()


def draw_progress_bar(window, row, current, total, label, width, color_pair_id):
    percentage = (current / total) * 100 if total > 0 else 0
    bar_length = width - len(label) - 15
    filled_length = int(bar_length * (current / total))
    bar = "█" * filled_length + "-" * (bar_length - filled_length)
    window.addstr(row, 2, f"{label}: [{bar}] {percentage:.0f}%", curses.color_pair(color_pair_id))

def draw_stats_window(stats_window, stats_dict, lock):
    stats_window.clear()
    stats_window.box()
    stats_window.addstr(1, 2, "Category Stats (Retrieved/Total)", curses.color_pair(2))
    
    with lock:
        row = 3
        max_y, max_x = stats_window.getmaxyx()
        for category, data in stats_dict.items():
            if row < max_y - 1:
                is_complete = data['retrieved'] >= data['total'] and data['total'] > 0
                color = curses.color_pair(1) if is_complete else curses.color_pair(3)
                stat_str = f"{category}: {data['retrieved']}/{data['total']}"
                if is_complete:
                    stat_str += " ✓"
                stats_window.addstr(row, 2, stat_str[:max_x-3], color)
                row += 1
            else:
                break
    stats_window.refresh()

def train_model(message_window, progress_window, time_window, script_start_time, width):
    add_scrollable_message(
        message_window,
        "\n--- Starting Model Training ---", curses.color_pair(2)
    )

    if not os.path.exists(DATA_FILE):
        add_scrollable_message(message_window, f"Error: Data file not found at {DATA_FILE}", curses.color_pair(4))
        return False, None

    try:
        df = pd.read_csv(DATA_FILE)
        df.dropna(subset=["message"], inplace=True)
        
        if df.empty:
            add_scrollable_message(message_window, "Error: Dataset is empty.", curses.color_pair(4))
            return False, None

        add_scrollable_message(message_window, "Fitting encoders (with forced vocabulary safety)...", curses.color_pair(3))
        
        main_label_encoder = LabelEncoder().fit(df["label"])
        
        is_security_event_label_encoder = LabelEncoder()
        is_security_event_label_encoder.fit(list(df["is_security_event"].astype(str).unique()) + ["True", "False"])
        
        known_priorities = ["none", "low", "medium", "high"]
        priority_label_encoder = LabelEncoder()
        priority_label_encoder.fit(list(df["priority"].astype(str).unique()) + known_priorities)
        
        event_type_label_encoder = LabelEncoder().fit(df["event_type"].astype(str))
        
        add_scrollable_message(message_window, "Fitted all label encoders.", curses.color_pair(1))

        # Create and fit a single, shared vectorizer
        add_scrollable_message(message_window, "Fitting the shared TfidfVectorizer...", curses.color_pair(3))
        shared_vectorizer = TfidfVectorizer(stop_words="english", max_features=5000)
        shared_vectorizer.fit(df["message"])
        add_scrollable_message(message_window, "Shared vectorizer fitted.", curses.color_pair(1))

        df_main_train = df.copy()
        label_counts = df_main_train['label'].value_counts()
        rare_classes = label_counts[label_counts == 1].index.tolist()
        if rare_classes:
            df_main_train = df_main_train[~df_main_train['label'].isin(rare_classes)]
            add_scrollable_message(message_window, f"Filtered out rare MAIN LABEL classes for training: {rare_classes}", curses.color_pair(4))

        X_main = df_main_train["message"]
        y_main_label = df_main_train["label"]
        
        main_model, _ = _train_model(
            X_main, y_main_label, "Main Label", shared_vectorizer, message_window, progress_window, width, main_label_encoder
        )

        X_full = df["message"]
        y_is_security_event = df["is_security_event"].astype(str)

        is_security_event_model, _ = _train_model(
            X_full, y_is_security_event, "Is Security Event", shared_vectorizer, message_window, progress_window, width, is_security_event_label_encoder
        )

        security_events_df = df[df["is_security_event"] == True].copy()
        
        event_type_model = None
        priority_model = None

        df_event_type_train = df.copy()
        event_type_counts = df_event_type_train["event_type"].value_counts()
        rare_event_classes = event_type_counts[event_type_counts == 1].index.tolist()

        if rare_event_classes:
            df_event_type_train = df_event_type_train[~df_event_type_train['event_type'].isin(rare_event_classes)]
            add_scrollable_message(message_window, f"Filtered out rare EVENT TYPE classes for training: {rare_event_classes}", curses.color_pair(4))

        if not df_event_type_train.empty:
            X_event_type = df_event_type_train["message"]
            y_event_type = df_event_type_train["event_type"]

            event_type_model, _ = _train_model(
                X_event_type, y_event_type, "Event Type", shared_vectorizer, message_window, progress_window, width, event_type_label_encoder
            )

        X_priority = df["message"]
        y_priority = df["priority"]
        
        priority_model, _ = _train_model(
            X_priority, y_priority, "Priority", shared_vectorizer, message_window, progress_window, width, priority_label_encoder
        )

        add_scrollable_message(message_window, "\nAll training processes finished. Starting ONNX conversion...", curses.color_pair(3))

        model_status = convert_to_onnx_models(
            shared_vectorizer, main_model, main_label_encoder,
            is_security_event_model, is_security_event_label_encoder,
            event_type_model, event_type_label_encoder,
            priority_model, priority_label_encoder,
            message_window,
        )

        generate_ensemble_config(message_window, model_status)

        add_scrollable_message(message_window, "\n--- Saving Joblib Artifacts (Encoders) ---", curses.color_pair(2))
        
        if main_model:
            _safe_save(shared_vectorizer, VECTORIZER_PATH)
            _safe_save(main_model, MODEL_PATH)
            _safe_save(main_label_encoder, LABEL_ENCODER_PATH)
        
        if is_security_event_model:
            _safe_save(is_security_event_model, IS_SECURITY_EVENT_MODEL_PATH)
            _safe_save(is_security_event_label_encoder, IS_SECURITY_EVENT_LABEL_ENCODER_PATH)

        if event_type_model:
            _safe_save(event_type_model, EVENT_TYPE_MODEL_PATH)
            _safe_save(event_type_label_encoder, EVENT_TYPE_LABEL_ENCODER_PATH)
            
        if priority_model:
            _safe_save(priority_model, PRIORITY_MODEL_PATH)
            _safe_save(priority_label_encoder, PRIORITY_LABEL_ENCODER_PATH)

        add_scrollable_message(message_window, "Joblib artifacts sync complete.", curses.color_pair(1))

        version_path = os.path.join(MODEL_DIR, "training_version.txt")
        training_time = datetime.now().isoformat()
        with open(version_path, "w") as f:
            f.write(training_time)
        add_scrollable_message(message_window, f"Wrote training version: {training_time}", curses.color_pair(1))

        return True, None

    except Exception as e:
        add_scrollable_message(message_window, f"  -> Error during model training: {e}", curses.color_pair(4))
        return False, e

def _safe_save(obj, path):
    if obj is None:
        return
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(obj, path)
    except Exception as e:
        pass

def _train_model(
    X,
    y,
    model_task_name,
    vectorizer, # Now passed in
    message_window,
    progress_window,
    width,
    label_encoder,
):
    add_scrollable_message(
        message_window,
        f"\n--- Training Model for: {model_task_name} ---",
        curses.color_pair(2),
    )

    y_encoded = label_encoder.transform(y).astype(np.int32)

    add_scrollable_message(
        message_window,
        f"Label distribution (for filtered data used in training {model_task_name}):\n{y.value_counts()}",
        curses.color_pair(3),
    )

    if y.nunique() < 2:
        add_scrollable_message(
            message_window,
            f"\nSkipping training for '{model_task_name}': Only one class found.",
            curses.color_pair(4)
        )
        return None, None

    X_train, X_test, y_train_encoded, y_test_encoded = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    progress_window.clear()
    progress_window.box()
    progress_window.addstr(1, 2, "Training Progress:", curses.color_pair(2))
    progress_window.addstr(3, 2, f"Vectorizing Text for {model_task_name}...", curses.color_pair(2))
    progress_window.refresh()

    # Use the provided vectorizer, don't fit it again
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    add_scrollable_message(message_window, f"Text vectorized for {model_task_name}.", curses.color_pair(3))

    progress_window.clear()
    progress_window.box()
    progress_window.addstr(1, 2, "Training Progress:", curses.color_pair(2))
    progress_window.addstr(3, 2, f"Training Model for {model_task_name}...", curses.color_pair(2))
    progress_window.refresh()

    if CUML_AVAILABLE:
        add_scrollable_message(message_window, "cuML detected. Using GPU for training...", curses.color_pair(1))
        model = cuml.LogisticRegression()
    else:
        add_scrollable_message(message_window, "Using scikit-learn for training...", curses.color_pair(2))
        model = LogisticRegression(random_state=42, class_weight="balanced", solver="saga", n_jobs=-1)

    model.fit(X_train_vec, y_train_encoded)
    add_scrollable_message(message_window, f"Model trained for {model_task_name}.", curses.color_pair(3))

    add_scrollable_message(message_window, f"\nEvaluating model performance for {model_task_name}...", curses.color_pair(3))
    y_pred_encoded = model.predict(X_test_vec)
    
    if CUML_AVAILABLE and hasattr(y_pred_encoded, 'get'):
        y_pred_encoded = y_pred_encoded.get()

    known_labels_int = model.classes_
    try:
        known_labels_str = label_encoder.inverse_transform(known_labels_int)
    except Exception:
        known_labels_str = [str(l) for l in known_labels_int]

    report = classification_report(y_test_encoded, y_pred_encoded, labels=known_labels_int, target_names=known_labels_str)
    add_scrollable_message(message_window, "Classification Report:", curses.color_pair(3))
    add_scrollable_message(message_window, report, curses.color_pair(3))

    add_scrollable_message(message_window, f"--- Model Training for {model_task_name} Finished (InMemory) ---", curses.color_pair(2))

    return model, label_encoder

def _generate_config_pbtxt(model_name, input_name, input_data_type, input_dims, output_configs):
    input_dims_str = f"[{', '.join(map(str, input_dims))}]"
    output_sections = []
    for output in output_configs:
        output_dims_str = f"[{', '.join(map(str, output['dims']))}]"
        output_sections.append(f"""
output [
  {{
    name: "{output['name']}"
    data_type: {output['data_type']}
    dims: {output_dims_str}
  }}
]"""
)

    config_content = f"""
name: "{model_name}"
platform: "onnxruntime_onnx"
max_batch_size: 0
version_policy {{ latest {{ num_versions: 1 }} }}
input [
  {{
    name: "{input_name}"
    data_type: {input_data_type}
    dims: {input_dims_str}
  }}
]
{chr(10).join(output_sections)}
"""
    return config_content

def convert_to_onnx_models(
    shared_vectorizer, main_model, main_label_encoder,
    is_security_event_model, is_security_event_label_encoder,
    event_type_model, event_type_label_encoder,
    priority_model, priority_label_encoder,
    message_window,
):
    add_scrollable_message(message_window, "\n--- Starting ONNX Model Conversion ---", curses.color_pair(2))

    ONNX_VECTORIZER_MODEL_DIR = os.path.join(MODEL_DIR, "vectorizer")
    ONNX_VECTORIZER_PATH = os.path.join(ONNX_VECTORIZER_MODEL_DIR, "1", "model.onnx")
    ONNX_MODEL_MODEL_DIR = os.path.join(MODEL_DIR, "model")
    ONNX_MODEL_PATH = os.path.join(ONNX_MODEL_MODEL_DIR, "1", "model.onnx")
    ONNX_IS_SECURITY_EVENT_MODEL_MODEL_DIR = os.path.join(MODEL_DIR, "is_security_event_model")
    ONNX_IS_SECURITY_EVENT_MODEL_PATH = os.path.join(ONNX_IS_SECURITY_EVENT_MODEL_MODEL_DIR, "1", "model.onnx")
    ONNX_EVENT_TYPE_MODEL_MODEL_DIR = os.path.join(MODEL_DIR, "event_type_model")
    ONNX_EVENT_TYPE_MODEL_PATH = os.path.join(ONNX_EVENT_TYPE_MODEL_MODEL_DIR, "1", "model.onnx")
    ONNX_PRIORITY_MODEL_MODEL_DIR = os.path.join(MODEL_DIR, "priority_model")
    ONNX_PRIORITY_MODEL_PATH = os.path.join(ONNX_PRIORITY_MODEL_MODEL_DIR, "1", "model.onnx")

    for p in [ONNX_VECTORIZER_PATH, ONNX_MODEL_PATH, ONNX_IS_SECURITY_EVENT_MODEL_PATH,
              ONNX_EVENT_TYPE_MODEL_PATH, ONNX_PRIORITY_MODEL_PATH]:
        os.makedirs(os.path.dirname(p), exist_ok=True)

    def _convert_to_onnx(skl_model, onnx_path, input_type, model_name_for_log, model_dir, label_encoder=None):
        add_scrollable_message(message_window, f"Converting {model_name_for_log}...", curses.color_pair(3))
        
        model_type_str = str(type(skl_model))
        
        if 'cuml' in model_type_str.lower() or (hasattr(skl_model, '__module__') and 'cuml' in skl_model.__module__.lower()):
            if not isinstance(skl_model, TfidfVectorizer):
                add_scrollable_message(message_window, f"  -> Detected cuML model ({model_name_for_log}). Bridging to Sklearn...", curses.color_pair(3))
                try:
                    from sklearn.linear_model import LogisticRegression as SklearnLR
                    
                    pure_model = SklearnLR()
                    
                    if hasattr(skl_model.classes_, 'to_numpy'):
                        pure_model.classes_ = skl_model.classes_.to_numpy()
                    elif hasattr(skl_model.classes_, 'get'):
                        pure_model.classes_ = skl_model.classes_.get()
                    else:
                        pure_model.classes_ = np.array(skl_model.classes_)
                        
                    if pure_model.classes_.dtype.kind in ('f', 'c'):
                        pure_model.classes_ = pure_model.classes_.astype(np.int32)

                    if hasattr(skl_model.coef_, 'get'):
                        pure_model.coef_ = skl_model.coef_.get()
                    else:
                        pure_model.coef_ = np.array(skl_model.coef_)
                        
                    if hasattr(skl_model.intercept_, 'get'):
                        pure_model.intercept_ = skl_model.intercept_.get()
                    else:
                        pure_model.intercept_ = np.array(skl_model.intercept_)

                    if hasattr(skl_model, 'n_features_in_'):
                        pure_model.n_features_in_ = skl_model.n_features_in_
                    else:
                        pure_model.n_features_in_ = pure_model.coef_.shape[1]

                    pure_model.n_iter_ = np.array([1], dtype=np.int32)
                    pure_model.solver = 'saga'

                    skl_model = pure_model
                    add_scrollable_message(message_window, f"  -> Bridge successful.", curses.color_pair(1))
                except Exception as bridge_e:
                     add_scrollable_message(message_window, f"  -> Bridge failed: {bridge_e}", curses.color_pair(4))

        try:
            if isinstance(skl_model, TfidfVectorizer):
                initial_type = [("input", StringTensorType([None, 1]))]
                onnx_model = convert_sklearn(skl_model, initial_types=initial_type, target_opset=11)
                
                add_scrollable_message(message_window, "  -> Converting to FP16...", curses.color_pair(3))
                onnx_model = float16.convert_float_to_float16(onnx_model)
                
                dummy_input = skl_model.transform(["dummy text"]).toarray()
                n_features = dummy_input.shape[1]
                output_configs = [{"name": "variable", "data_type": "TYPE_FP16", "dims": [-1, n_features]}]
                input_data_type = "TYPE_STRING"
                input_dims = [-1, 1]
            else:
                initial_type = input_type
                onnx_model = convert_sklearn(skl_model, initial_types=initial_type, target_opset=11, options={"zipmap": False})
                
                add_scrollable_message(message_window, "  -> Converting to FP16...", curses.color_pair(3))
                onnx_model = float16.convert_float_to_float16(onnx_model)
                
                num_classes = len(skl_model.classes_)
                output_configs = [
                    {"name": "label", "data_type": "TYPE_INT64", "dims": [-1]},
                    {"name": "probabilities", "data_type": "TYPE_FP16", "dims": [-1, num_classes]}
                ]
                input_data_type = "TYPE_FP16"
                input_dims = [-1 if d is None else d for d in input_type[0][1].shape]

            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            config_content = _generate_config_pbtxt(os.path.basename(model_dir), "input", input_data_type, input_dims, output_configs)
            with open(os.path.join(model_dir, "config.pbtxt"), "w") as f:
                f.write(config_content)
        except Exception as e:
            add_scrollable_message(message_window, f"Error converting {model_name_for_log}: {e}", curses.color_pair(4))

    if shared_vectorizer:
        _convert_to_onnx(shared_vectorizer, ONNX_VECTORIZER_PATH, None, "Shared Vectorizer", ONNX_VECTORIZER_MODEL_DIR)
        n_features = shared_vectorizer.transform(["dummy"]).shape[1]
        input_type = [("input", FloatTensorType([None, n_features]))]

        if main_model:
            _convert_to_onnx(main_model, ONNX_MODEL_PATH, input_type, "Main Model", ONNX_MODEL_MODEL_DIR, main_label_encoder)
        if is_security_event_model:
            _convert_to_onnx(is_security_event_model, ONNX_IS_SECURITY_EVENT_MODEL_PATH, input_type, "Sec Model", ONNX_IS_SECURITY_EVENT_MODEL_MODEL_DIR, is_security_event_label_encoder)
        if event_type_model:
            _convert_to_onnx(event_type_model, ONNX_EVENT_TYPE_MODEL_PATH, input_type, "Type Model", ONNX_EVENT_TYPE_MODEL_MODEL_DIR, event_type_label_encoder)
        if priority_model:
            _convert_to_onnx(priority_model, ONNX_PRIORITY_MODEL_PATH, input_type, "Prio Model", ONNX_PRIORITY_MODEL_MODEL_DIR, priority_label_encoder)

    add_scrollable_message(message_window, "\n--- ONNX Conversion Finished ---", curses.color_pair(2))
    
    return {
        "has_type": bool(event_type_model),
        "has_prio": bool(priority_model)
    }

def generate_ensemble_config(message_window, model_status):
    add_scrollable_message(message_window, "\n--- Generating Ensemble Config ---", curses.color_pair(2))
    
    ENSEMBLE_MODEL_DIR = os.path.join(MODEL_DIR, "cyber_log_pipeline")
    os.makedirs(os.path.join(ENSEMBLE_MODEL_DIR, "1"), exist_ok=True)
    CONFIG_PATH = os.path.join(ENSEMBLE_MODEL_DIR, "config.pbtxt")

    config_content = """
name: "cyber_log_pipeline"
platform: "ensemble"
max_batch_size: 0
version_policy { latest { num_versions: 1 } }

input [
  {
    name: "input_message"
    data_type: TYPE_STRING
    dims: [ -1, 1 ]
  }
]

output [
  {
    name: "label"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "is_security_event"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }"""

    if model_status["has_type"]:
        config_content += """,
  {
    name: "event_type"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }"""
    
    if model_status["has_prio"]:
        config_content += """,
  {
    name: "priority"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }"""

    config_content += """
]

ensemble_scheduling {
  step [
    {
      model_name: "vectorizer"
      model_version: -1
      input_map { key: "input" value: "input_message" }
      output_map { key: "variable" value: "feature_vector" }
    },
    {
      model_name: "model"
      model_version: -1
      input_map { key: "input" value: "feature_vector" }
      output_map { key: "label" value: "label" }
    },
    {
      model_name: "is_security_event_model"
      model_version: -1
      input_map { key: "input" value: "feature_vector" }
      output_map { key: "label" value: "is_security_event" }
    }"""

    if model_status["has_type"]:
        config_content += """,
    {
      model_name: "event_type_model"
      model_version: -1
      input_map { key: "input" value: "feature_vector" }
      output_map { key: "label" value: "event_type" }
    }"""

    if model_status["has_prio"]:
        config_content += """,
    {
      model_name: "priority_model"
      model_version: -1
      input_map { key: "input" value: "feature_vector" }
      output_map { key: "label" value: "priority" }
    }"""

    config_content += """
  ]
}
"""
    try:
        with open(CONFIG_PATH, "w") as f:
            f.write(config_content)
        add_scrollable_message(message_window, f"SUCCESS: Ensemble config created at {CONFIG_PATH}", curses.color_pair(1))
    except Exception as e:
        add_scrollable_message(message_window, f"CRITICAL ERROR: Could not write config: {e}", curses.color_pair(4))

def ncurses_main(stdscr, time_from, time_to, limit):
    stdscr.clear()
    stdscr.refresh()

    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_GREEN, -1)
    curses.init_pair(2, curses.COLOR_YELLOW, -1)
    curses.init_pair(3, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_RED, curses.COLOR_BLACK)

    height, width = stdscr.getmaxyx()
    progress_window_height = 7
    time_window_height = 3
    stats_window_height = len(QUERIES) + 4
    
    min_height = progress_window_height + time_window_height + stats_window_height + 5
    if height < min_height:
        stdscr.addstr(0, 0, f"Terminal too small. Minimum height required: {min_height} rows.", curses.color_pair(4))
        if height > 1:
            stdscr.addstr(1, 0, "Please resize your terminal and try again. Press any key to exit.", curses.color_pair(4))
        stdscr.refresh()
        stdscr.getch()
        return

    progress_window = curses.newwin(progress_window_height, width, 0, 0)
    time_window = curses.newwin(time_window_height, width, progress_window_height, 0)
    stats_window = curses.newwin(stats_window_height, width, progress_window_height + time_window_height, 0)
    message_window = curses.newwin(
        height - progress_window_height - time_window_height - stats_window_height,
        width,
        progress_window_height + time_window_height + stats_window_height,
        0,
    )
    message_window.scrollok(True)

    with UI_LOCK:
        progress_window.box()
        time_window.box()
        stats_window.box()
        message_window.box()
        progress_window.addstr(1, 2, "Initializing...", curses.color_pair(3))
        progress_window.refresh()
        time_window.refresh()
        stats_window.refresh()
        message_window.refresh()

    stats_lock = threading.Lock()
    category_stats = {
        category: {"total": 0, "retrieved": 0} for category in QUERIES
    }
    timing_stats = []
    timing_lock = threading.Lock()
    effective_total_rows = [0]
    effective_total_lock = threading.Lock()

    all_data = []
    total_queries = sum(len(category_data["queries"]) for category_data in QUERIES.values())
    completed_queries = 0
    script_start_time = datetime.now()

    stdscr.timeout(100)

    fetcher_tasks = []
    for category_name, category_data in QUERIES.items():
        current_label = category_data["label"]
        current_is_security_event = category_data["is_security_event"]
        current_event_type = category_data["event_type"]
        current_priority = category_data["priority"]
        
        for query in category_data["queries"]:
            fetcher = LogFetcher(
                query,
                category_name,
                current_label,
                current_is_security_event,
                current_event_type,
                current_priority,
                message_window,
                category_stats,
                stats_lock,
                timing_stats,
                timing_lock,
                time_from,
                time_to,
                limit,
                effective_total_rows,
                effective_total_lock
            )
            fetcher_tasks.append(fetcher)

    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_fetcher = {executor.submit(task.run): task for task in fetcher_tasks}
        not_done = set(future_to_fetcher.keys())

        try:
            while not_done:
                done, not_done = wait(not_done, timeout=0.1)

                for future in done:
                    fetcher = future_to_fetcher[future]
                    try:
                        future.result()
                        if fetcher.status == "completed":
                            df = fetcher.result_df
                            if not df.empty:
                                all_data.append(df)
                            fetcher._add_message_to_window(
                                "\n" + "-" * (fetcher.max_x - 2), curses.color_pair(3)
                            )
                        else:
                            fetcher._add_message_to_window(
                                f"  -> Error during fetch: {fetcher.error}", curses.color_pair(4)
                            )
                    except Exception as exc:
                        add_scrollable_message(
                            message_window,
                            f"  -> A query generated an exception: {exc}",
                            curses.color_pair(4),
                        )
                    completed_queries += 1

                with UI_LOCK:
                    progress_window.clear()
                    progress_window.box()
                    draw_progress_bar(progress_window, 1, completed_queries, total_queries, "Overall Progress", width, 1)
                    progress_window.addstr(3, 2, f"Collecting Data... ({completed_queries}/{total_queries})", curses.color_pair(2))
                    progress_window.refresh()

                    draw_stats_window(stats_window, category_stats, stats_lock)

                    etc_str = "ETC: Calculating..."
                    with timing_lock:
                        if timing_stats:
                            total_rows_processed = sum(rows for duration, rows in timing_stats)
                            total_time_spent = sum(duration for duration, rows in timing_stats)
                            
                            if total_time_spent > 0:
                                average_rate = total_rows_processed / total_time_spent
                                
                                with effective_total_lock:
                                    total_rows_to_fetch = effective_total_rows[0]
                                
                                with stats_lock:
                                    total_rows_retrieved = sum(cat['retrieved'] for cat in category_stats.values())

                                rows_remaining = total_rows_to_fetch - total_rows_retrieved
                                
                                if average_rate > 0 and rows_remaining > 0:
                                    etc_seconds = int(rows_remaining / average_rate)
                                    hours, rem = divmod(etc_seconds, 3600)
                                    minutes, seconds = divmod(rem, 60)
                                    etc_str = f"ETC: {hours:02d}:{minutes:02d}:{seconds:02d}"
                                else:
                                    etc_str = "ETC: --:--:--"
                            else:
                                etc_str = "ETC: --:--:--"

                    current_time = datetime.now()
                    elapsed_time = current_time - script_start_time
                    time_window.clear()
                    time_window.box()
                    start_str = f"Start: {script_start_time.strftime('%H:%M:%S')}"
                    total_seconds_int = int(elapsed_time.total_seconds())
                    hours, minutes, seconds = total_seconds_int // 3600, (total_seconds_int % 3600) // 60, total_seconds_int % 60
                    elapsed_str = f"Elapsed: {hours:02d}:{minutes:02d}:{seconds:02d}"
                    
                    time_display_str = f"{start_str} | {elapsed_str} | {etc_str}"
                    time_window.addstr(1, 2, time_display_str[: width - 4], curses.color_pair(3))
                    time_window.refresh()

                key = stdscr.getch()
                if key == ord("q"):
                    for future in not_done:
                        future.cancel()
                    executor.shutdown(wait=False)
                    return
        except KeyboardInterrupt:
            add_scrollable_message(message_window, "\nInterrupted by user. Shutting down...", curses.color_pair(4))
            for future in not_done:
                future.cancel()
            return

    if all_data:
        add_scrollable_message(message_window, "\n--- Managing Data Lifecycle ---", curses.color_pair(2))
        
        # 1. Create the new dataframe from fetched data
        new_df = pd.concat(all_data, ignore_index=True)
        
        # 2. Load existing data if it exists
        if os.path.exists(DATA_FILE):
            try:
                existing_df = pd.read_csv(DATA_FILE)
                add_scrollable_message(message_window, f"Loaded {len(existing_df)} existing rows.", curses.color_pair(3))
                
                # Combine existing and new
                combined_df = pd.concat([existing_df, new_df], ignore_index=True)
            except Exception as e:
                add_scrollable_message(message_window, f"Error reading existing CSV: {e}. Starting fresh.", curses.color_pair(4))
                combined_df = new_df
        else:
            combined_df = new_df

        # 3. Deduplicate based on ID or Timestamp+Message
        # Assuming 'message' + 'timestamp' is a unique enough key if you don't have a UUID
        initial_count = len(combined_df)
        combined_df.drop_duplicates(subset=['timestamp', 'message', 'source'], keep='last', inplace=True)
        dropped_count = initial_count - len(combined_df)
        add_scrollable_message(message_window, f"Deduplicated: Removed {dropped_count} overlapping rows.", curses.color_pair(3))

        # 4. Prune Old Data (Sliding Window - e.g., keep last 60 days)
        # Ensure timestamp is datetime
        try:
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], format='mixed', utc=True)
            retention_date = datetime.now(timezone.utc) - timedelta(days=60) # ADJUST THIS SETTING
            combined_df = combined_df[combined_df['timestamp'] >= retention_date]
            add_scrollable_message(message_window, f"Pruning: Kept data after {retention_date.date()}", curses.color_pair(3))
        except Exception as e:
            add_scrollable_message(message_window, f"Warning: Could not prune by date (timestamp parsing failed): {e}", curses.color_pair(2))

        # 5. Save and Train
        add_scrollable_message(message_window, f"Saving {len(combined_df)} rows to {DATA_FILE}...", curses.color_pair(3))
        combined_df.to_csv(DATA_FILE, index=False)
        
        train_model(message_window, progress_window, time_window, script_start_time, width)
    else:
        # If no NEW data was found, we might still want to retrain if we have a CSV? 
        # Or just exit.
        add_scrollable_message(message_window, "\nNo new data collected.", curses.color_pair(4))

    add_scrollable_message(message_window, "\nPress any key to exit.", curses.color_pair(2))
    stdscr.getch()


def main():
    """Parses command-line arguments and starts the application."""
    global USERNAME, PASSWORD
    parser = argparse.ArgumentParser(description="Graylog data extractor and model trainer.")
    parser.add_argument(
        "--time",
        default="1d",
        help="Set the time range for queries. Examples: '1h' (1 hour), '3d' (3 days), '2w' (2 weeks). Default is '1d'."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of rows to retrieve per query."
    )
    args = parser.parse_args()

    try:
        time_delta = parse_time_delta(args.time)
    except ValueError as e:
        print(f"Error: {e}")
        return

    time_to = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    time_from = (datetime.now(timezone.utc) - time_delta).strftime("%Y-%m-%dT%H:%M:%S.000Z")
    
    USERNAME = input("Enter Graylog username: ")
    PASSWORD = getpass.getpass("Enter Graylog password/token: ")
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    curses.wrapper(ncurses_main, time_from, time_to, args.limit)

if __name__ == "__main__":
    main()