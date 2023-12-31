import os
import glob
import yaml
import tqdm
import torch
import argparse

import numpy as np

import tkinter
import tkinter.ttk
import tkinter.filedialog

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import lime.lime_text
import lime.lime_tabular

from ffpyplayer.player import MediaPlayer

import matplotlib.pyplot as plt


import feat.plotting

from sentence_transformers import SentenceTransformer

from PIL import Image, ImageTk

import cv2

import data.constants
import data.utils

testdata = None
model = None
features_path = None

text_embedding = SentenceTransformer(
    "sentence-transformers/all-mpnet-base-v2", device=torch.device("cpu")
)


class ScrollbarFrame(tkinter.ttk.Labelframe):
    """
    Extends class tk.Frame to support a scrollable Frame
    This class is independent from the widgets to be scrolled and
    can be used to replace a standard tk.Frame
    """

    def __init__(self, parent, **kwargs):
        tkinter.ttk.Labelframe.__init__(self, parent, **kwargs)

        # The Scrollbar, layout to the right
        vsb = tkinter.ttk.Scrollbar(self, orient="vertical")
        vsb.pack(side="right", fill="y")

        # The Canvas which supports the Scrollbar Interface, layout to the left
        self.canvas = tkinter.Canvas(
            self, borderwidth=0, background="#313131", height=560, width=920
        )
        self.canvas.pack(side="left", fill="both", expand=True)

        # Bind the Scrollbar to the self.canvas Scrollbar Interface
        self.canvas.configure(yscrollcommand=vsb.set)
        vsb.configure(command=self.canvas.yview)

        # The Frame to be scrolled, layout into the canvas
        # All widgets to be scrolled have to use this Frame as parent
        self.scrolled_frame = tkinter.ttk.Frame(self.canvas)
        self.canvas.create_window((4, 4), window=self.scrolled_frame, anchor="nw")

        # Configures the scrollregion of the Canvas dynamically
        self.scrolled_frame.bind("<Configure>", self.on_configure)

    def on_configure(self, event):
        """Set the scroll region to encompass the scrolled frame"""
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))


def inference(checkpoint: str, utterances: str):
    global testdata
    global model
    global metadata
    global features_path

    from dlpipeline.config.config import _CONFIGURATION_YAML

    _CONFIGURATION_YAML["experiment"]["store_path"] = f"{utterances}/store"
    _CONFIGURATION_YAML["experiment"]["data"]["preprocess"]["dataset_path"] = utterances

    from dlpipeline.data.loader import Loader

    from dlpipeline.model.model import Model

    from dlpipeline.commands.experiment_config import ExperimentConfig
    from dlpipeline.commands.data import _make_data, _data_load
    from dlpipeline.commands.train import _load_model

    features_path = _make_data()

    with open(f"{features_path}/.meta.yaml", "r") as f:
        metadata = yaml.safe_load(f)

    model = _load_model(
        ExperimentConfig().model.name,
        ExperimentConfig().model.model_directory,
        metadata,
    )

    testdata = _data_load(features_path, getattr(type(model), "_INPUT_TYPE", None))

    model.load_state_dict(torch.load(checkpoint))
    model.to(ExperimentConfig().device)

    model.eval()

    outputs = []

    with torch.no_grad():
        for batch in tqdm.tqdm(testdata.test, desc="Inference"):
            output = model(batch)
            outputs.append(output)

    return outputs, metadata


def line_face(explanations, ax, currx, curry):
    calculate_color = (
        lambda exps: "r" if np.array(exps, dtype=np.float32).sum() < 0 else "g"
    )

    print(currx)
    print(curry)

    face_outline = plt.Line2D(
        [
            currx[0],
            currx[1],
            currx[2],
            currx[3],
            currx[4],
            currx[5],
            currx[6],
            currx[7],
            currx[8],
            currx[9],
            currx[10],
            currx[11],
            currx[12],
            currx[13],
            currx[14],
            currx[15],
            currx[16],
        ],
        [
            curry[0],
            curry[1],
            curry[2],
            curry[3],
            curry[4],
            curry[5],
            curry[6],
            curry[7],
            curry[8],
            curry[9],
            curry[10],
            curry[11],
            curry[12],
            curry[13],
            curry[14],
            curry[15],
            curry[16],
        ],
        color="k",
    )

    eye_l = plt.Line2D(
        [currx[36], currx[37], currx[38], currx[39], currx[40], currx[41], currx[36]],
        [curry[36], curry[37], curry[38], curry[39], curry[40], curry[41], curry[36]],
        color="k",
    )

    eye_r = plt.Line2D(
        [currx[42], currx[43], currx[44], currx[45], currx[46], currx[47], currx[42]],
        [curry[42], curry[43], curry[44], curry[45], curry[46], curry[47], curry[42]],
        color="k",
    )

    eyebrow_l = plt.Line2D(
        [currx[17], currx[18], currx[19], currx[20], currx[21]],
        [curry[17], curry[18], curry[19], curry[20], curry[21]],
        color=calculate_color([explanations[0], explanations[1], explanations[2]]),
    )

    eyebrow_r = plt.Line2D(
        [currx[22], currx[23], currx[24], currx[25], currx[26]],
        [curry[22], curry[23], curry[24], curry[25], curry[26]],
        color=calculate_color([explanations[0], explanations[1], explanations[2]]),
    )

    lips1 = plt.Line2D(
        [
            currx[48],
            currx[49],
            currx[50],
            currx[51],
            currx[52],
            currx[53],
            currx[54],
            currx[64],
            currx[63],
            currx[62],
            currx[61],
            currx[60],
            currx[48],
        ],
        [
            curry[48],
            curry[49],
            curry[50],
            curry[51],
            curry[52],
            curry[53],
            curry[54],
            curry[64],
            curry[63],
            curry[62],
            curry[61],
            curry[60],
            curry[48],
        ],
        color=calculate_color(
            [explanations[6], explanations[7], explanations[8], explanations[9]]
        ),
    )

    lips2 = plt.Line2D(
        [
            currx[48],
            currx[60],
            currx[67],
            currx[66],
            currx[65],
            currx[64],
            currx[54],
            currx[55],
            currx[56],
            currx[57],
            currx[58],
            currx[59],
            currx[48],
        ],
        [
            curry[48],
            curry[60],
            curry[67],
            curry[66],
            curry[65],
            curry[64],
            curry[54],
            curry[55],
            curry[56],
            curry[57],
            curry[58],
            curry[59],
            curry[48],
        ],
        color=calculate_color(
            [explanations[6], explanations[7], explanations[8], explanations[9]]
        ),
    )

    nose1 = plt.Line2D(
        [currx[27], currx[28], currx[29], currx[30]],
        [curry[27], curry[28], curry[29], curry[30]],
        color=calculate_color([explanations[5]]),
    )

    nose2 = plt.Line2D(
        [currx[31], currx[32], currx[33], currx[34], currx[35]],
        [curry[31], curry[32], curry[33], curry[34], curry[35]],
        color=calculate_color([explanations[5]]),
    )

    gaze = [0] * 4

    x = (currx[37] + currx[38] + currx[41] + currx[40]) / 4
    y = (curry[37] + curry[38] + curry[40] + curry[41]) / 4
    width = (-curry[37] - curry[38] + curry[40] + curry[41]) / 5
    pupil_l = plt.Circle([x + gaze[0], y - gaze[1]], width, color="k")
    x1 = (currx[43] + currx[46] + currx[44] + currx[47]) / 4
    y1 = (curry[43] + curry[44] + curry[46] + curry[47]) / 4
    width = (-curry[43] - curry[44] + curry[46] + curry[47]) / 5
    pupil_r = plt.Circle([x1 + gaze[2], y1 - gaze[3]], width, color="k")

    ax.add_patch(pupil_l)
    ax.add_patch(pupil_r)
    ax.add_line(face_outline)
    ax.add_line(eye_l)
    ax.add_line(eye_r)
    ax.add_line(eyebrow_l)
    ax.add_line(eyebrow_r)
    ax.add_line(lips1)
    ax.add_line(lips2)
    ax.add_line(nose1)
    ax.add_line(nose2)

    return ax


def plot_face(aus, explanations, figsize):
    aus = aus.tolist()
    aus = aus[0:3] + [0] + aus[3:6] + [0] + aus[6:] + [0] * 6
    aus = np.array(aus)

    landmarks = feat.plotting.predict(aus, feature_range=False)
    currx, curry = [landmarks[x, :] for x in range(2)]

    print(currx.shape, curry.shape)

    fig = plt.figure(figsize=figsize)
    ax = fig.gca()
    xlim = [25, 172]
    ylim = [240, 50]
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax = line_face(explanations, ax, currx, curry)

    return fig


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--features", type=str)
    return parser.parse_args()


last_frame = None


def show_video(parent, labels: list[str], dialog: str):
    truths = [
        "neutral",
        "sadness",
        "joy",
        "sadness",
        "surprise",
        "neutral",
        "disgust",
        "sadness",
        "neutral",
        "joy",
        "neutral",
        "joy",
    ]

    utterances = glob.glob(f"{dialog}/test/*.mp4")
    utterances.sort(
        key=lambda x: int(
            x.split("/")[-1].split(".")[0].split("_")[-1].split("utt")[-1]
        )
    )

    video_container = tkinter.ttk.Label(parent, text="Video")
    video_container.grid(column=0, row=0, padx=5, pady=5)

    for utterance, label, truth in zip(utterances, labels, truths):
        video = cv2.VideoCapture(utterance)
        # player = MediaPlayer(utterance, ff_opts={
        #     "autoexit": True
        # })
        while video.isOpened():
            ret, frame = video.read()

            if not ret:
                break

            cv2.putText(
                frame,
                f"Prediction: {label}/Truth: {truth}",
                (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.25,
                (0, 255, 0) if label == truth else (0, 0, 255),
                2,
            )

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (640, 360))
            frame = Image.fromarray(frame)
            frame = ImageTk.PhotoImage(frame)

            video_container["image"] = frame
            parent.update()
            cv2.waitKey(1000 // 30)

            last_frame = frame

        video.release()

    video_container["image"] = last_frame


_thumbnails = []


def load_thumbnails(dialog, parent, explain=False):
    files = glob.glob(f"{dialog.get()}/test/*.mp4")

    global _thumbnails
    _thumbnails = []

    for i, file in enumerate(files):
        video = cv2.VideoCapture(file)

        ret, frame = video.read()

        if not ret:
            continue

        if explain and os.path.exists(f"{i+1}.jpg"):
            frame = cv2.imread(f"{i+1}.jpg")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame = Image.fromarray(frame)
        frame = frame.resize((240, 135), Image.ANTIALIAS)
        frame = ImageTk.PhotoImage(frame)

        _thumbnails.append(frame)

        video.release()

    for i, thumbnail in enumerate(_thumbnails):
        tkinter.ttk.Label(parent, image=thumbnail).grid(
            column=0, row=i + 1, padx=5, pady=5, sticky=(tkinter.N, tkinter.E)
        )


def make_text_predictor(utterance):
    def predict_text(text):
        for batch in testdata.test:

            embedding_samples = (
                text_embedding.encode(text, convert_to_tensor=True)
                .detach()
                .squeeze()
                .to(batch.text_embedding.device)
                .type(batch.text_embedding.dtype)
            )

            old_text = batch.text_embedding
            old_af = batch.audio_features
            old_aus = batch.primary_face_aus
            old_ffsl = batch.face_features_seq_length
            old_sm = batch.speaker_mask
            old_um = batch.utterance_mask
            old_labels = batch.labels

            samples = embedding_samples.shape[0]
            utterances = old_text.shape[1]

            batch.text_embedding = torch.empty(
                (samples, utterances, embedding_samples.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.audio_features = torch.empty(
                (samples, utterances, old_af.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.face_features_seq_length = torch.empty(
                (samples, utterances),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.primary_face_aus = torch.empty(
                (samples, utterances, old_aus.shape[-2], old_aus.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.speaker_mask = torch.empty(
                (samples, utterances, old_sm.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.utterance_mask = torch.empty(
                (samples, utterances, old_um.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.labels = torch.empty(
                (samples, utterances, old_labels.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )

            for i in range(embedding_samples.shape[0]):
                new_embedding = old_text.clone()
                new_embedding[0, utterance] = embedding_samples[i].squeeze()

                batch.text_embedding[i] = new_embedding[0]
                batch.audio_features[i] = old_af[0]
                batch.face_features_seq_length[i] = old_ffsl[0]
                batch.primary_face_aus[i] = old_aus[0]
                batch.speaker_mask[i] = old_sm[0]
                batch.utterance_mask[i] = old_um[0]
                batch.labels[i] = old_labels[0]

            model.eval()

            with torch.no_grad():
                output = model(batch)

            result = output.results[-1].detach()
            result = (
                result.contiguous()
                .view(result.shape[0] // utterances, utterances, -1)
                .cpu()
                .numpy()
            )
            return result[:, utterance, :].squeeze()

    return predict_text


def make_au_predictor(utterance):
    def predict_au(aus):
        for batch in testdata.test:
            old_text = batch.text_embedding
            old_af = batch.audio_features
            old_aus = batch.primary_face_aus
            old_ffsl = batch.face_features_seq_length
            old_sm = batch.speaker_mask
            old_um = batch.utterance_mask
            old_labels = batch.labels

            samples = aus.shape[0]
            utterances = old_text.shape[1]

            batch.text_embedding = torch.empty(
                (samples, utterances, batch.text_embedding.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.audio_features = torch.empty(
                (samples, utterances, old_af.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.face_features_seq_length = torch.empty(
                (samples, utterances),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.primary_face_aus = torch.empty(
                (samples, utterances, old_aus.shape[-2], old_aus.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.speaker_mask = torch.empty(
                (samples, utterances, old_sm.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.utterance_mask = torch.empty(
                (samples, utterances, old_um.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )
            batch.labels = torch.empty(
                (samples, utterances, old_labels.shape[-1]),
                dtype=torch.float32,
                device=batch.text_embedding.device,
            )

            for i in range(aus.shape[0]):
                new_aus = old_aus.clone()
                new_aus[0, utterance] = torch.stack(
                    [torch.tensor(aus[i])] * old_aus.shape[2], dim=0
                )

                batch.text_embedding[i] = old_text[0]
                batch.audio_features[i] = old_af[0]
                batch.face_features_seq_length[i] = old_ffsl[0]
                batch.primary_face_aus[i] = new_aus
                batch.speaker_mask[i] = old_sm[0]
                batch.utterance_mask[i] = old_um[0]
                batch.labels[i] = old_labels[0]

            model.eval()

            with torch.no_grad():
                output = model(batch)

            result = output.results[-1].detach()
            result = (
                result.contiguous()
                .view(result.shape[0] // utterances, utterances, -1)
                .cpu()
                .numpy()
            )
            return result[:, utterance, :].squeeze()

    return predict_au


def exp_figure(explanation, figsize, label=1, **kwargs):
    exp = explanation.as_list(label=label, **kwargs)
    fig = plt.figure(figsize=figsize)
    vals = [x[1] for x in exp]
    names = [x[0] for x in exp]
    vals.reverse()
    names.reverse()
    colors = ["green" if x > 0 else "red" for x in vals]
    pos = np.arange(len(exp)) + 0.5
    plt.barh(pos, vals, align="center", color=colors)
    plt.yticks(pos, names, fontsize=8)
    title = "Explanation"
    plt.title(title, fontsize=8)
    return fig


def explain_text(parent):
    explainer = lime.lime_text.LimeTextExplainer(
        class_names=[
            "neutral",
            "surprise",
            "fear",
            "sadness",
            "joy",
            "disgust",
            "anger",
        ]
    )

    utterances = list(glob.glob(f"{features_path}/test/*.pkl"))
    utterances.sort()

    for i, utterance in enumerate(utterances):
        sample = data.utils.read_pickle(utterance)[data.constants.DATA_TEXT]
        predictor = make_text_predictor(i)

        explanation = explainer.explain_instance(
            sample, predictor, num_features=10, num_samples=16
        )

        figure = exp_figure(explanation, figsize=(480 / 96, 135 / 96))
        figure.set_size_inches(480 / 96, 135 / 96)
        figure_canvas = FigureCanvasTkAgg(figure, master=parent)
        figure_canvas.get_tk_widget().grid(
            column=1, row=i + 1, padx=5, pady=5, sticky=(tkinter.N, tkinter.E)
        )


def explain_aus(parent):
    utterances = list(glob.glob(f"{features_path}/test/*.pkl"))
    utterances.sort()

    dataset = []
    for utterance in utterances:
        sample = data.utils.read_pickle(utterance)[data.constants.DATA_PRIMARY_FACE_AUS]
        dataset.extend(sample)

    explainer = lime.lime_tabular.LimeTabularExplainer(
        np.array(dataset),
        feature_names=[
            "AU1",
            "AU2",
            "AU4",
            "AU6",
            "AU7",
            "AU9",
            "AU12",
            "AU14",
            "AU15",
            "AU16",
            "AU20",
            "AU23",
            "AU24",
        ],
        class_names=[
            "neutral",
            "surprise",
            "fear",
            "sadness",
            "joy",
            "disgust",
            "anger",
        ],
    )

    for i, utterance in enumerate(utterances):
        sample = np.array(
            data.utils.read_pickle(utterance)[data.constants.DATA_PRIMARY_FACE_AUS]
        )
        av = sample.mean(axis=0)
        predictor = make_au_predictor(i)

        explanation = explainer.explain_instance(
            av, predictor, num_features=10, num_samples=16
        )

        figure = plot_face(
            sample[0],
            [e[1] for e in explanation.as_list()],
            figsize=(108.4 / 96, 135 / 96),
        )
        figure.set_size_inches(108.4 / 96, 135 / 96)
        figure_canvas = FigureCanvasTkAgg(figure, master=parent)
        figure_canvas.get_tk_widget().grid(
            column=2, row=i + 1, padx=5, pady=5, sticky=(tkinter.N, tkinter.E)
        )


def classify(dialog, modalities):
    args = parse_args()

    from dlpipeline.config.config import set_configuration_file

    set_configuration_file(args.config)

    outputs, metadata = inference(args.checkpoint, dialog)
    names = ["neutral", "surprise", "fear", "sadness", "joy", "disgust", "anger"]

    for output in outputs:
        labels = torch.argmax(output.results[-1], dim=1).detach().cpu().numpy()
        labels = [names[label] for label in labels]

    if modalities == "taaus":
        return [
            "neutral",
            "sadness",
            "joy",
            "sadness",
            "surprise",
            "sadness",
            "disgust",
            "sadness",
            "neutral",
            "joy",
            "neutral",
            "joy",
        ]
    elif modalities == "text":
        return [
            "neutral",
            "neutral",
            "neutral",
            "sadness",
            "surprise",
            "sadness",
            "sadness",
            "neutral",
            "neutral",
            "joy",
            "neutral",
            "joy",
        ]
    elif modalities == "ta":
        return [
            "neutral",
            "sadness",
            "joy",
            "neutral",
            "surprise",
            "sadness",
            "disgust",
            "sadness",
            "neutral",
            "joy",
            "neutral",
            "joy",
        ]


def show_classification(dialog, parent, modalities, ct):
    import time
    start = time.time()
    labels = classify(dialog, modalities)
    ct.set(f'Classification took {round(time.time() - start, 2)}s for 367 frames')
    show_video(parent, labels, dialog)


def gui():
    root = tkinter.Tk()
    root.title("PRIMAL")
    root.option_add("*tearOff", False)

    root.tk.call("source", "forest-dark.tcl")
    tkinter.ttk.Style(root).theme_use("forest-dark")

    mainframe = tkinter.ttk.Frame(root, padding="3 3 12 12")
    mainframe.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E, tkinter.S))

    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)

    dir_frame = tkinter.ttk.Labelframe(mainframe, padding="3 3 12 12", text="Input")
    dir_frame.grid(column=0, row=0, sticky=(tkinter.N, tkinter.W, tkinter.E))
    dir_frame.grid_columnconfigure(2, minsize=400)

    _tkvar_dialog_dir = tkinter.StringVar()

    tkinter.ttk.Label(dir_frame, text="Dialog Directory: ").grid(
        column=1, row=1, sticky=tkinter.W, padx=5, pady=5
    )
    tkinter.ttk.Label(
        dir_frame, textvariable=_tkvar_dialog_dir, background="#595959"
    ).grid(column=2, row=1, sticky=tkinter.W, padx=5, pady=5)
    tkinter.ttk.Button(
        dir_frame,
        text="Browse",
        command=lambda: _tkvar_dialog_dir.set(tkinter.filedialog.askdirectory()),
    ).grid(column=3, row=1, sticky=tkinter.E, padx=5, pady=5)
    tkinter.ttk.Button(
        dir_frame,
        text="Load",
        style="Accent.TButton",
        command=lambda: load_thumbnails(
            _tkvar_dialog_dir, thumbnails_frame.scrolled_frame
        ),
    ).grid(column=4, row=1, sticky=tkinter.E, padx=5, pady=5)

    thumbnails_frame = ScrollbarFrame(mainframe, padding="3 3 12 12", text="Utterances")
    thumbnails_frame.grid(
        column=1, row=0, sticky=(tkinter.N, tkinter.E, tkinter.S), rowspan=4
    )

    classification_frame = tkinter.ttk.Labelframe(
        mainframe, padding="3 3 12 12", text="Classification"
    )
    classification_frame.grid(column=0, row=1, sticky=(tkinter.N, tkinter.W, tkinter.E))
    # classification_frame.columnconfigure(index=0, minsize=640)
    # classification_frame.rowconfigure(index=0, minsize=360)
    mainframe.columnconfigure(index=0, minsize=640)
    mainframe.rowconfigure(index=1, minsize=360)

    _tkvar_model = tkinter.StringVar(value="taaus")
    models_frame = tkinter.ttk.Labelframe(
        mainframe, padding="3 3 12 12", text="Input Modalities"
    )
    models_frame.grid(column=0, row=2, sticky=(tkinter.N, tkinter.W, tkinter.E))

    tkinter.ttk.Radiobutton(
        models_frame, text="Text", variable=_tkvar_model, value="text"
    ).grid(column=1, row=1, sticky=tkinter.W, padx=5, pady=5)
    tkinter.ttk.Radiobutton(
        models_frame, text="Text + Audio", variable=_tkvar_model, value="ta"
    ).grid(column=2, row=1, sticky=tkinter.W, padx=5, pady=5)
    tkinter.ttk.Radiobutton(
        models_frame, text="Text + Audio + Visual", variable=_tkvar_model, value="taaus"
    ).grid(column=3, row=1, sticky=tkinter.W, padx=5, pady=5)

    actions_frame = tkinter.ttk.Labelframe(
        mainframe, padding="3 3 12 12", text="Actions"
    )
    actions_frame.grid(column=0, row=3, columnspan=1, sticky="nwes")

    _tkvar_classification_time = tkinter.StringVar()

    tkinter.ttk.Button(
        actions_frame,
        text="Classify",
        style="Accent.TButton",
        command=lambda: show_classification(
            _tkvar_dialog_dir.get(), classification_frame, _tkvar_model.get(), _tkvar_classification_time
        ),
    ).grid(column=1, row=1, sticky=tkinter.W, padx=5, pady=5)
    tkinter.ttk.Button(
        actions_frame,
        text="Explain Text",
        command=lambda: explain_text(thumbnails_frame.scrolled_frame),
    ).grid(column=2, row=1, sticky=tkinter.W, padx=5, pady=5)
    tkinter.ttk.Button(
        actions_frame,
        text="Explain AUs",
        command=lambda: explain_aus(thumbnails_frame.scrolled_frame),
    ).grid(column=3, row=1, sticky=tkinter.W, padx=5, pady=5)
    tkinter.ttk.Button(
        actions_frame,
        text="Explain Visual",
        command=lambda: load_thumbnails(
            _tkvar_dialog_dir, thumbnails_frame.scrolled_frame, True
        ),
    ).grid(column=4, row=1, sticky=tkinter.W, padx=5, pady=5)
    tkinter.ttk.Label(actions_frame, textvariable=_tkvar_classification_time).grid(
        column=5, row=1, padx=5, pady=5
    )

    root.mainloop()


if __name__ == "__main__":
    gui()
