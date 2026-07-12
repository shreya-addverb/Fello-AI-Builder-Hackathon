import {useEffect,useState} from 'react';
import {useLocation} from 'react-router-dom';
import {AlertTriangle,ArrowRight,Braces,Building2,Code2,Play,RotateCcw,Users} from 'lucide-react';
import {analyze} from '../lib/api';
import {useApp} from '../context/app-context';
import {Card,JsonViewer} from '../components/ui';
import {PipelineList} from '../components/pipeline';
import {Report} from './report';

export default function Analysis(){
  const location=useLocation();
  const type=location.pathname.includes('visitor')?'visitor':'company';
  const [company,setCompany]=useState('');
  const [domain,setDomain]=useState('');
  const [visitor,setVisitor]=useState('');
  const [error,setError]=useState('');
  const {result,setResult,runStatus,setRunStatus,addLog,developer}=useApp();

  useEffect(()=>setError(''),[type]);

  async function run(){
    setError('');setResult(null);setRunStatus('running');
    addLog('INFO',`${type==='company'?'Company':'Visitor'} analysis request started`);
    try{
      const body=type==='company'
        ?{company_name:company.trim()||undefined,domain:domain.trim()||undefined}
        :JSON.parse(visitor);
      const data=await analyze(type,body);
      setResult(data);setRunStatus(data.pipeline_status);
      addLog(data.pipeline_status==='completed'?'SUCCESS':'WARNING',`Pipeline returned with status ${data.pipeline_status}`);
    }catch(caught){
      const message=caught instanceof Error?caught.message:'Unexpected analysis error';
      setError(message);setRunStatus('failed');addLog('ERROR',message);
    }
  }

  const ready=type==='company'?Boolean(company.trim()||domain.trim()):Boolean(visitor.trim());
  return <div className="page analysis-page">
    <div className="page-heading"><div>
      <p className="kicker"><span/> {type} intelligence</p>
      <h1>{type==='company'?'Research an account':'Identify a website visitor'}</h1>
      <p>{type==='company'?'Enter a company name and optional domain for evidence-based account research.':'Paste visitor activity to identify the account and infer buying intent.'}</p>
    </div>{result&&<button className="button secondary" onClick={()=>{setResult(null);setRunStatus('idle')}}><RotateCcw/> New analysis</button>}</div>

    {!result&&runStatus!=='running'&&<Card className="input-card">
      <div className="input-mode"><div className="mode-icon">{type==='company'?<Building2/>:<Users/>}</div><div>
        <h2>{type==='company'?'Company details':'Visitor event JSON'}</h2>
        <p>{type==='company'?'Adding a domain improves identification and enrichment.':'Provide the visitor fields required by the API.'}</p>
      </div></div>
      {type==='company'
        ?<div className="company-fields">
          <label className="field"><span>Company</span><input value={company} onChange={event=>setCompany(event.target.value)} placeholder="Company name" autoFocus onKeyDown={event=>event.key==='Enter'&&ready&&run()}/></label>
          <label className="field"><span>Domain</span><input value={domain} onChange={event=>setDomain(event.target.value)} placeholder="example.com" onKeyDown={event=>event.key==='Enter'&&ready&&run()}/></label>
        </div>
        :<label className="field"><span>Visitor payload</span><textarea value={visitor} onChange={event=>setVisitor(event.target.value)} rows={13} spellCheck={false} placeholder={'{\n  "visitor_id": "...",\n  "ip": "...",\n  "pages_visited": []\n}'}/></label>}
      <button className="button primary analyze-button" disabled={!ready} onClick={run}><Play/> Analyze {type}<ArrowRight/></button>
    </Card>}

    {error&&<Card className="error-card">
      <div className="error-title"><AlertTriangle/><div><p className="eyebrow">Analysis failed</p><h3>The request could not be completed.</h3></div></div>
      <p>{error}</p>
      {developer&&<JsonViewer data={{type,message:error,endpoint:`/analyze/${type}`}}/>}
      <button className="button secondary" onClick={run}><RotateCcw/> Retry</button>
    </Card>}

    {runStatus==='running'&&<div className="run-layout"><Card>
      <div className="running-head"><div className="pulse-icon"><Code2/></div><div><p className="eyebrow">Pipeline in progress</p><h2>Researching account…</h2><p>Results will appear when the backend completes the synchronous pipeline.</p></div></div>
      <PipelineList status="running" completed={[]}/>
    </Card><Card className="request-card"><p className="eyebrow">Active request</p><h3>Waiting for backend</h3><div className="request-row"><span>Method</span><b>POST</b></div><div className="request-row"><span>Endpoint</span><code>/analyze/{type}</code></div><p className="notice"><Braces/> Stage streaming is not exposed by the backend.</p></Card></div>}
    {result&&<Report result={result}/>} 
  </div>
}
