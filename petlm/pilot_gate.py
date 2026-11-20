"""Read-only corpus gates; pending candidates never count as approved data."""
from collections import Counter
import hashlib
import json
from pathlib import Path


def checksum(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream,'sha256').hexdigest()


def registered_rows(register_path, tokenizer):
    register=json.loads(Path(register_path).read_text(encoding='utf8'))
    catalog={}
    for source in register['sources']:
        folder=Path(source['path'])
        if checksum(folder/'manifest.json')!=source['manifest_sha256']:
            raise ValueError('Changed approved source manifest')
        meta=json.loads((folder/'manifest.json').read_text(encoding='utf8'))
        for name,digest in meta['files'].items():
            if checksum(folder/name)!=digest:raise ValueError('Changed approved source file')
        approval_path=folder/'user-review-record.json'
        if checksum(approval_path)!=source['approval_sha256']:raise ValueError('Changed approval record')
        approval=json.loads(approval_path.read_text(encoding='utf8'))
        if approval['manifest_sha256']!=source['manifest_sha256'] or approval['decision']!='pass':
            raise ValueError('Approval does not bind this source')
        if approval['review_json_sha256']!=checksum(folder/'review.json'):
            raise ValueError('Approval does not bind review text')
        decisions={r['id']:r for r in approval['rows']}
        for row in json.loads((folder/'review.json').read_text(encoding='utf8'))['rows']:
            if row['id'] not in decisions:continue
            decision=decisions[row['id']]
            digest=hashlib.sha256(row['text'].encode()).hexdigest()
            if decision['decision']!='pass' or digest!=decision['text_sha256']:
                raise ValueError('Row not approved or approved text changed')
            if row['tokens_with_eos']!=len(tokenizer.encode(row['text']))+1:
                raise ValueError('Incorrect approved token count')
            catalog[(folder.name,row['id'])]=dict(row,human_approved=True,approval_path=str(approval_path))
    rows=[]
    for item in register['rows']:
        row=catalog[(item['candidate_source'],item['id'])]
        if item['human_decision']!='pass' or item['text_sha256']!=hashlib.sha256(row['text'].encode()).hexdigest():
            raise ValueError('Register approval mismatch')
        if item['split']!=row['split'] or item['tokens_with_eos']!=row['tokens_with_eos']:
            raise ValueError('Register split or token mismatch')
        rows.append(dict(row,candidate_source=item['candidate_source']))
    if dict(Counter(r['split'] for r in rows))!=register['counts']:
        raise ValueError('Register counts mismatch')
    return rows


def summary(rows):
    sources=Counter();topics=Counter();documents=Counter();source_rows=Counter();topic_rows=Counter();doc_rows=Counter()
    for row in rows:
        source=row['coverage_source_group'];topic=row['coverage_topic'];doc=row['document_group'];tokens=row['tokens_with_eos']
        sources[source]+=tokens;topics[topic]+=tokens;documents[doc]+=tokens
        source_rows[source]+=1;topic_rows[topic]+=1;doc_rows[doc]+=1
    total=sum(sources.values())
    return {'rows':len(rows),'tokens_with_eos':total,'source_groups':len(sources),
            'topic_groups':len([t for t in topics if t!='unclassified']),'documents':len(documents),
            'source_tokens':dict(sources),'topic_tokens':dict(topics),'source_rows':dict(source_rows),'topic_rows':dict(topic_rows),
            'max_source_token_share':max(sources.values(),default=0)/max(total,1),
            'max_document_token_share':max(documents.values(),default=0)/max(total,1),
            'max_blocks_per_document':max(doc_rows.values(),default=0)}


def assess(rows, protocol, snapshot_frozen=False):
    approved=[r for r in rows if r.get('human_approved') is True]
    train=summary([r for r in approved if r['split']=='train'])
    validation=summary([r for r in approved if r['split']=='validation' and r.get('primary_independent_validation') is True])
    g=protocol['data_gates'];checks={}
    for field,limit in [('tokens_with_eos','minimum_train_tokens_including_eos'),('source_groups','minimum_train_source_groups'),('topic_groups','minimum_train_topic_groups')]:
        checks[limit]=train[field]>=g[limit]
    checks['maximum_train_source_token_share']=train['max_source_token_share']<=g['maximum_train_source_token_share']
    checks['maximum_train_document_token_share']=train['max_document_token_share']<=g['maximum_train_document_token_share']
    checks['minimum_tokens_each_train_topic']=all(train['topic_tokens'].get(t,0)>=g['minimum_tokens_each_train_topic'] for t in g['allowed_topics'])
    for field,limit in [('tokens_with_eos','minimum_independent_validation_tokens_including_eos'),('rows','minimum_independent_validation_rows'),('documents','minimum_independent_validation_documents'),('source_groups','minimum_independent_validation_source_groups'),('topic_groups','minimum_independent_validation_topics')]:
        checks[limit]=validation[field]>=g[limit]
    checks['minimum_rows_each_validation_source']=bool(validation['source_rows']) and all(n>=g['minimum_rows_each_validation_source'] for n in validation['source_rows'].values())
    checks['minimum_tokens_each_validation_source']=bool(validation['source_tokens']) and all(n>=g['minimum_tokens_each_validation_source'] for n in validation['source_tokens'].values())
    checks['minimum_rows_each_validation_topic']=all(validation['topic_rows'].get(t,0)>=g['minimum_rows_each_validation_topic'] for t in g['allowed_topics'])
    checks['minimum_tokens_each_validation_topic']=all(validation['topic_tokens'].get(t,0)>=g['minimum_tokens_each_validation_topic'] for t in g['allowed_topics'])
    checks['maximum_validation_source_token_share']=validation['max_source_token_share']<=g['maximum_validation_source_token_share']
    checks['maximum_validation_blocks_per_document']=validation['max_blocks_per_document']<=g['maximum_validation_blocks_per_document']
    checks['independent_validation_authors_not_in_training']=not ({r['coverage_source_group'] for r in approved if r['split']=='train'}&{r['coverage_source_group'] for r in approved if r['split']=='validation' and r.get('primary_independent_validation')})
    checks['final_data_snapshot_frozen']=snapshot_frozen is True
    return {'training_ready':all(checks.values()),'train':train,'primary_independent_validation':validation,
            'excluded_pending_rows':len(rows)-len(approved),'legacy_diagnostic_validation_rows':sum(r['split']=='validation' and not r.get('primary_independent_validation') for r in approved),
            'checks':checks,'failed_conditions':[k for k,v in checks.items() if not v]}
