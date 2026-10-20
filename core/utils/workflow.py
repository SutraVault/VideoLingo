"""Persistent output plans and revision-aware task reuse (no UI dependencies)."""
from dataclasses import dataclass,asdict,fields
from pathlib import Path
import hashlib,json,threading,uuid,shutil
from datetime import datetime


@dataclass
class Options:
    original: bool = True
    dubbing: bool = False
    original_subtitles: str = 'translation'
    dub_subtitles: str = 'translation'
    review_translation: bool = True
    review_terms: bool = False
    auto_start: bool = True
    voice_revision: str = ''
    translation_revision: str = ''

    @classmethod
    def from_dict(cls,data):
        result=cls(**{f.name:data[f.name] for f in fields(cls) if f.name in data})
        if result.original_subtitles not in ('translation','bilingual'):
            raise ValueError('Invalid original-video subtitle mode')
        if result.dub_subtitles not in ('translation','bilingual','none'):
            raise ValueError('Invalid dubbing-video subtitle mode')
        return result


FILES={'asr':'log/cleaned_chunks.xlsx','nlp':'log/split_by_nlp.txt',
       'meaning':'log/split_by_meaning.txt','terms':'log/terminology.json',
       'translation':'log/translation_results.xlsx','subtitles':'trans.srt',
       'original':'output_sub.mp4','dubbing':'output_dub.mp4','audio':'dub.mp3'}


def digest(path):
    path=Path(path)
    if not path.is_file():return ''
    h=hashlib.sha256()
    try:
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(1024*1024),b''):h.update(chunk)
    except OSError:return ''  # Excel may briefly replace/lock a file while saving.
    return h.hexdigest()


def plan(options,facts,approved):
    """Return executable stages up to the next explicit human boundary."""
    stages=[]
    for stage in ('asr','nlp','meaning','terms'):
        if not facts.get(stage):stages.append(stage)
    if options.review_terms and not approved.get('terms'):
        return stages,'waiting_terms'
    if not facts.get('translation'):stages.append('translation')
    if options.review_translation and not approved.get('translation'):
        return stages,'waiting_translation'
    if not facts.get('subtitles'):stages.extend(['split','subtitles'])
    if options.original and not facts.get('original'):stages.append('original')
    if options.dubbing and not facts.get('dubbing'):
        if not facts.get('audio'):stages.extend(['audio_tasks','references','synthesis','merge_audio'])
        stages.append('dubbing')
    return stages,'completed'


class Project:
    def __init__(self,source,output='output',options=None):
        self.source=Path(source).resolve();self.output=Path(output).resolve()
        self.path=self.output/'workflow_state.json';self.lock=threading.RLock()
        stat=self.source.stat()
        self.identity=hashlib.sha256(f'{self.source}:{stat.st_size}:{stat.st_mtime_ns}'.encode()).hexdigest()
        try:self.state=json.loads(self.path.read_text(encoding='utf-8'))
        except (OSError,ValueError):self.state={}
        if self.state.get('source_id') and self.state['source_id']!=self.identity:
            self.archive(list(FILES.values())+['audio','gpt_log','workflow_state.json',
                'src.srt','src_trans.srt','trans_src.srt',
                'log/translation_results_for_subtitles.xlsx','log/translation_results_remerged.xlsx',
                'log/translation_results_proofread.xlsx'])
        if self.state.get('source_id')!=self.identity:
            self.state={'source_id':self.identity,'source':str(self.source),'status':'ready',
                        'options':asdict(options) if options else {},'approved':{},'artifacts':{},'last_error':''}
            # Adopt pre-existing work once. Later edits are detected by hashes.
            self.state['translation_inputs']=self.translation_inputs()
            self.state['terms_revision']=self.options.translation_revision
            for stage in ('subtitles','original','dubbing','audio'):
                if (self.output/FILES[stage]).is_file():self.mark_artifact(stage,save=False)
            self.save()
        elif self.state.get('status')=='running':
            # New process/session registry: never silently restart interrupted work.
            self.state['status']='interrupted';self.save()

    @property
    def options(self):return Options.from_dict(self.state.get('options',{}))

    def set_options(self,options):
        self.state['options']=asdict(options);self.save()

    def save(self):
        with self.lock:
            self.output.mkdir(parents=True,exist_ok=True)
            temporary=self.path.with_suffix('.'+uuid.uuid4().hex+'.tmp')
            temporary.write_text(json.dumps(self.state,ensure_ascii=False,indent=2),encoding='utf-8')
            temporary.replace(self.path)

    def revision(self):return digest(self.output/FILES['translation'])

    def translation_inputs(self):
        return {'terms':digest(self.output/FILES['terms']),'asr':digest(self.output/FILES['asr']),
                'settings':self.options.translation_revision}

    def signature(self,stage):
        payload={'translation':self.revision()}
        if stage=='original':payload['subtitles']=self.options.original_subtitles
        if stage=='dubbing':payload['subtitles']=self.options.dub_subtitles
        if stage in ('dubbing','audio'):payload['voice']=self.options.voice_revision
        return payload

    def mark_artifact(self,stage,save=True):
        self.state.setdefault('artifacts',{})[stage]={'signature':self.signature(stage),'path':FILES[stage]}
        if save:self.save()

    def facts(self):
        facts={stage:(self.output/path).is_file() for stage,path in FILES.items()}
        facts['terms']=facts['terms'] and self.state.get('terms_revision','')==self.options.translation_revision
        facts['translation']=facts['translation'] and self.state.get('translation_inputs')==self.translation_inputs()
        for stage in ('subtitles','original','dubbing','audio'):
            metadata=self.state['artifacts'].get(stage,{})
            facts[stage]=facts[stage] and metadata.get('signature')==self.signature(stage)
        # An incomplete subtitle bundle must be rebuilt before TTS.
        facts['subtitles']=facts['subtitles'] and all((self.output/p).is_file() for p in
            ('src.srt','audio/src_subs_for_audio.srt','audio/trans_subs_for_audio.srt'))
        for prerequisite,dependents in (
            ('asr',('nlp','meaning','terms','translation','subtitles','original','audio','dubbing')),
            ('terms',('translation','subtitles','original','audio','dubbing')),
            ('translation',('subtitles','original','audio','dubbing'))):
            if not facts[prerequisite]:
                for dependent in dependents:facts[dependent]=False
        return facts

    def approvals(self):
        return {kind:bool(digest(self.output/FILES[kind])) and self.state['approved'].get(kind)==digest(self.output/FILES[kind])
                for kind in ('terms','translation')}

    def approve(self,kind):
        value=digest(self.output/FILES[kind])
        if not value:raise ValueError('Nothing ready for review')
        self.state['approved'][kind]=value;self.save()

    def next_plan(self):return plan(self.options,self.facts(),self.approvals())

    def archive(self,paths):
        """Move only explicitly selected derived files, retaining every prior version."""
        candidates=[self.output/p for p in paths if (self.output/p).exists()]
        if not candidates:return
        destination=self.output/'versions'/(datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:6])
        for source in candidates:
            source=source.resolve()
            if not source.is_relative_to(self.output):raise ValueError('Archive outside output directory')
            target=destination/source.relative_to(self.output)
            target.parent.mkdir(parents=True,exist_ok=True)
            shutil.move(str(source),str(target))
            for meta in self.state['artifacts'].values():
                if meta.get('path')==str(source.relative_to(self.output)).replace('\\','/'):
                    meta['path']=str(target.relative_to(self.output)).replace('\\','/')
        self.save()

    def prepare(self,stage):
        if stage=='terms' and not self.facts()['terms']:
            self.archive(['log/terminology.json'])
        elif stage=='translation' and not self.facts()['translation']:
            self.archive(['log/translation_results.xlsx','log/translation_results_proofread.xlsx'])
        elif stage=='split':
            self.archive(['log/translation_results_for_subtitles.xlsx','log/translation_results_remerged.xlsx',
                          'src.srt','trans.srt','src_trans.srt','trans_src.srt',
                          'audio/src_subs_for_audio.srt','audio/trans_subs_for_audio.srt'])
        elif stage=='original':self.archive([FILES['original']])
        elif stage=='audio_tasks':
            if self.state.get('prepared_audio')!=self.signature('audio'):
                self.archive(['audio/tts_tasks.xlsx','audio/tmp','audio/segs','dub.mp3','dub.srt','normalized_dub.wav'])
                self.state['prepared_audio']=self.signature('audio');self.save()
        elif stage=='dubbing':self.archive([FILES['dubbing']])

    def wrap(self,stage,operation):
        def execute():
            self.state.update(status='running',current=stage,current_status='running',last_error='');self.save()
            try:
                self.prepare(stage);operation()
                self.state['current_status']='completed';self.save()
                expected=FILES.get(stage) or (FILES['audio'] if stage=='merge_audio' else None)
                if expected and not (self.output/expected).is_file():
                    raise RuntimeError('Expected output was not generated: '+expected)
                if stage=='translation':self.state['translation_inputs']=self.translation_inputs();self.save()
                if stage=='terms':self.state['terms_revision']=self.options.translation_revision;self.save()
                if stage=='merge_audio':self.mark_artifact('audio')
                if stage in ('subtitles','original','dubbing'):
                    if not (self.output/FILES[stage]).is_file():raise RuntimeError('Expected output was not generated')
                    self.mark_artifact(stage)
            except Exception as exc:
                self.state.update(status='error',last_error=str(exc),failure_id=uuid.uuid4().hex);self.save();raise
        return execute

    def finish(self,boundary):
        self.state.update(status=boundary,current='',last_error='');self.save()
