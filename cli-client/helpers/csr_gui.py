import os
from rich.live import Live
from rich.panel import Panel
from rich.layout import Layout
from rich.padding import Padding
from rich.console import Console, RenderableType
from rich.progress import Progress, TextColumn, BarColumn, TimeRemainingColumn

GUI_NUM_REFRESH_PER_SECOND = 4
GUI_PARTIAL_RESULT_TAG = "<PARTIAL>"
GUI_FINAL_RESULT_TAG = "<FINAL>"
GUI_PROGRESS_BAR_TASK_DESCRIPTION_RUNNING = "Sending streaming audio messages"
GUI_PROGRESS_BAR_TASK_DESCRIPTION_FINISHED = "[bold green]All audio sent successfully!"


class CsrGUI:
    def __init__(self):
        self.__init_gui_layout()
        self.__set_logging_panel()
        self.__set_transcript_panel()
        self.__set_progress_bar()

    def __init_gui_layout(self):
        self._layout = Layout(name="root")
        self._layout.split(
            Layout(name="status", ratio=3),
            Layout(name="transcript", minimum_size=10),
        )
        self._layout["status"].split_column(
            Layout(name="progress", size=3),
            Layout(name="logging"),
        )

    def __set_logging_panel(self) -> RenderableType:
        self._logging_console = LogPanel()
        logging_panel = Panel(self._logging_console, title="Logs", border_style="green")
        self._layout["logging"].update(logging_panel)

    def __set_transcript_panel(self) -> RenderableType:
        self._transcript_console = TranscriptPanel()
        transcript_panel = Panel(self._transcript_console, title="Transcript", border_style="blue", padding=(2, 2))
        self._layout["transcript"].update(transcript_panel)

    def __set_progress_bar(self) -> RenderableType:
        self._progress_bar = Progress(
            TextColumn("{task.description}", justify="right"),
            BarColumn(bar_width=None),
            TextColumn("{task.completed} / {task.total} samples"),
            "[progress.percentage]({task.percentage:>3.1f}%)",
            TimeRemainingColumn(),
        )
        progress_panel = Padding(self._progress_bar, pad=(1, 1))
        self._layout["progress"].update(progress_panel)

    def start(self):
        self._screen = Live(self._layout, refresh_per_second=GUI_NUM_REFRESH_PER_SECOND, screen=False)
        self._screen.start(refresh=True)

    def stop(self):
        self._screen.stop()

    def start_progress_bar_task(self, total_audio_samples: int):
        self._streaming_task = self._progress_bar.add_task(
            GUI_PROGRESS_BAR_TASK_DESCRIPTION_RUNNING, total=total_audio_samples
        )

    def advance_progress_bar(self, advance: int):
        if not self._progress_bar.finished:
            self._progress_bar.update(self._streaming_task, advance=advance)
        if self._progress_bar.finished:
            self._progress_bar.update(self._streaming_task, description=GUI_PROGRESS_BAR_TASK_DESCRIPTION_FINISHED)
        self._screen.refresh()

    def add_partial_transcript(self, transcript: str):
        self._transcript_console.print(GUI_PARTIAL_RESULT_TAG, end='')
        self._transcript_console.print(transcript, end='')
        self._screen.refresh()

    def add_final_transcript(self, transcript: str):
        self._transcript_console.print(GUI_FINAL_RESULT_TAG, end='')
        self._transcript_console.print(transcript)
        self._screen.refresh()


class LogPanel(Console):
    def __init__(self, *args, **kwargs):
        console_file = open(os.devnull, 'w')
        super().__init__(record=True, file=console_file, *args, **kwargs)
        self.stored_logs = []

    def __rich_console__(self, console, options):
        logs = self.export_text(clear=True)
        clean_logs = [log for log in logs.split('\n') if log]
        total_logs = self.stored_logs + clean_logs
        self.stored_logs = total_logs[-options.height:]
        for line in self.stored_logs:
            yield line


class TranscriptPanel(Console):
    def __init__(self, *args, **kwargs):
        console_file = open(os.devnull, 'w')
        super().__init__(record=True, file=console_file, *args, **kwargs)
        self.stored_transcripts = []
        self.in_partial_line = False
        self.last_partial_line = ""

    def __rich_console__(self, console, options):
        temporary_transcripts = []
        transcripts = self.export_text(clear=True)
        
        for line in transcripts.split('\n'):
            if GUI_FINAL_RESULT_TAG in line:
                final_line = line.split(GUI_FINAL_RESULT_TAG)[-1]
                self.stored_transcripts.append(final_line)
                self.last_partial_line = ""
                self.in_partial_line = False
            elif GUI_PARTIAL_RESULT_TAG in line:
                temporary_line = line.split(GUI_PARTIAL_RESULT_TAG)[-1]
                temporary_transcripts.append(temporary_line)
                self.last_partial_line = temporary_line
                self.in_partial_line = True

        if self.in_partial_line and len(temporary_transcripts) == 0:
            temporary_transcripts = [self.last_partial_line]

        total_transcripts = self.stored_transcripts + temporary_transcripts
        yield "".join(total_transcripts)[-(options.height * options.max_width):]