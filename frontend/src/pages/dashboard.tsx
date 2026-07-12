import {Activity,ArrowRight,BrainCircuit,CheckCircle2,Clock3,Layers3,Network,Server} from 'lucide-react';
import {Link} from 'react-router-dom';
import {useQuery} from '@tanstack/react-query';
import {health} from '../lib/api';
import {Card,MetricCard,StatusBadge} from '../components/ui';
import {stages} from '../components/pipeline';

export default function Dashboard(){
  const query=useQuery({queryKey:['health'],queryFn:({signal})=>health(signal),refetchInterval:30000,retry:1});
  return <div className="page">
    <section className="hero">
      <div>
        <span className="kicker"><span/> AI account intelligence</span>
        <h1>Turn account signals into<br/>sales-ready intelligence.</h1>
        <p>Analyze a company or visitor activity through an evidence-based research pipeline.</p>
        <div className="hero-actions">
          <Link className="button primary" to="/analyze/company">Analyze a company <ArrowRight/></Link>
          <Link className="button secondary" to="/analyze/visitor">Analyze a visitor</Link>
        </div>
      </div>
      <Card className="architecture">
        <div className="card-head"><div><p className="eyebrow">Architecture</p><h3>Intelligence pipeline</h3></div><StatusBadge status={query.data?.status||'offline'}/></div>
        <div className="flow"><div><Network/><span>Input signal</span></div><i/><div><BrainCircuit/><span>AI research</span></div><i/><div><CheckCircle2/><span>Account report</span></div></div>
      </Card>
    </section>
    <section className="metrics">
      <MetricCard label="API status" value={query.isLoading?'Checking…':query.data?.status||'Offline'} detail={query.data?`${query.data.latency}ms latency`:'Backend unavailable'} icon={Activity}/>
      <MetricCard label="Pipeline" value="Ready" detail="Sequential orchestration" icon={Layers3}/>
      <MetricCard label="Providers" value="3" detail="Gemini · Tavily · Firecrawl" icon={Server}/>
      <MetricCard label="Stages" value={stages.length} detail="Identification to outreach" icon={Clock3}/>
    </section>
  </div>
}
