"""Assemble the existing editable Figure 1/2 layouts using real checkpoint outputs."""
from pathlib import Path
import argparse
import json
import shutil
import subprocess
import sys
import zipfile

import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
PAPER = REPO.parent/'6a89bc13108d28a2448f410a'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', default=str(REPO/'reports/paper_figures_arm7_4k'))
    p.add_argument('--out', default=str(PAPER/'Figs/arm7_4k'))
    args = p.parse_args()
    source, out = Path(args.source), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    manifest = {'kind': 'checkpoint predictions and measured simulator execution', 'assets': {}, 'generations': {}}
    for number, demo in [(1, 'teaser_demo'), (2, 'method_demo')]:
        dest = out/f'figure{number}'
        dest.mkdir(exist_ok=True)
        shutil.copytree(PAPER/'Figs'/demo/'assets', dest/'assets', dirs_exist_ok=True)
        old = json.loads((PAPER/'Figs'/demo/'assets_manifest.json').read_text())
        for name, record in old['assets'].items():
            manifest['assets'][f'figure{number}/assets/{name}'] = record

    def copy_asset(run_name, relative, dest, info):
        run = source/run_name
        metadata = json.loads((run/'generation.json').read_text())
        manifest['generations'][run_name] = metadata
        shutil.copyfile(run/relative, out/dest)
        manifest['assets'][dest] = dict(source=str(run/relative), checkpoint=metadata['checkpoint'],
                                      episode=metadata['episode'], template=metadata['template'], **info)

    for modality, stem in [('video','rgb'),('depth','depth'),('normal','normal'),('segmentation','roles')]:
        run = f'figure1_{modality}'
        copy_asset(run, 'view1/gt_perception/000.png', f'figure1/assets/{stem}_anchor.png',
                   dict(type='observed ground truth', source_frame=12, view='view1'))
        for slot, i, fi in [('future1',15,57), ('future2',29,99)]:
            for stream, filename in [('perception', f'{stem}_{slot}.png'), ('action',f'{stem}_action_{slot}.png')]:
                copy_asset(run, f'view1/{stream}/{i:03d}.png', f'figure1/assets/{filename}',
                           dict(type='model prediction', prediction_index=i, corresponding_source_frame=fi, view='view1'))
    for slot, i, fi in [('anchor',0,12), ('future1',15,57), ('future2',29,99)]:
        stream = 'gt_action' if i==0 else 'action'
        copy_asset('figure1_video',f'view1/{stream}/{i:03d}.png',f'figure1/assets/action_{slot}.png',
                   dict(type='observed ground truth' if i==0 else 'model prediction', prediction_index=i, corresponding_source_frame=fi, view='view1'))
    for slot, i, fi in [('future1',7,77), ('future2',14,98)]:
        for kind, stream in [('depth','perception'), ('action','action')]:
            copy_asset('figure2_depth',f'view1/{stream}/{i:03d}.png',f'figure2/assets/output_{kind}_{slot}.png',
                       dict(type='model prediction', prediction_index=i, corresponding_source_frame=fi, view='view1'))
    for view in [1,2]:
        for kind, stream in [('depth','perception'), ('action','action')]:
            copy_asset('figure2_depth',f'view{view}/gt_{stream}/000.png',f'figure2/assets/view{view}_{kind}_future1.png',
                       dict(type='observed ground truth / training-clip illustration', source_frame=56, view=f'view{view}'))

    # Use actual executed simulator images, never the generated video, for control.
    results, executed = {}, {}
    for model in ['arm0','arm7']:
        report = source/'rollouts'/f'rollout_figure_{model}_4k_close_drawer.json'
        record = json.loads(report.read_text())['results'][0]
        assert record.get('sim_video'), record
        assert not record.get('gt_replay')
        raw = Path(record['sim_video']).with_suffix('.npz')
        executed[model] = np.load(raw)['frames']
        results[model] = dict(record, lossless_source=str(raw))
    assert results['arm0']['scene_seed'] == results['arm7']['scene_seed']
    assert results['arm0']['prompt'] == results['arm7']['prompt']
    assert np.array_equal(executed['arm0'][0], executed['arm7'][0]), 'Initial observations must match exactly.'
    for model, name in [('arm0','control_rgb_only.png'),('arm7','control_xgenact.png')]:
        frames = executed[model]
        frame = frames[-1]
        Image.fromarray(frame[:, :frame.shape[1]//2]).save(out/'figure1/assets'/name)
        manifest['assets'][f'figure1/assets/{name}'] = dict(type='executed simulator rollout', model=model,
              source=results[model]['lossless_source'], frame_index=len(frames)-1, view='view1',
              success=results[model]['success'], scene_seed=results[model]['scene_seed'])
    initial = executed['arm7'][0]
    Image.fromarray(initial[:, :initial.shape[1]//2]).save(out/'figure1/assets/control_input.png')
    manifest['assets']['figure1/assets/control_input.png'] = dict(type='observed simulator RGB; identical for both policies',
          source=results['arm7']['lossless_source'], frame_index=0, scene_seed=results['arm7']['scene_seed'],view='view1')
    manifest['rollouts'] = results
    (out/'assets_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

    # Keep the original demos intact. Each output folder has its own rebuild script.
    builder = (PAPER/'Figs/teaser_demo/build_figure.py').read_text()
    substitutions = {
      'discussion draft using dataset placeholders':'arm7 step 4000 model generations',
      'All images are dataset ground truth, not predictions or policy rollouts.':'Left outputs are model predictions; right outputs are actual paired simulator executions.',
      'reserves space for a matched RGB control comparison':'shows a paired RGB control example',
      'PLACEHOLDER DEMO':'ARM7 · STEP 4000',
      "'DEMO ONLY'":"'ARM7 · 4K'",
      'Dataset frames and recorded actions fill all output slots; replace with predictions and policy rollouts.':'Left: generated futures. Right: actual execution in one paired scene; not an aggregate success-rate estimate.',
      "238,'#BDC6D3','GT'":"238,'#BDC6D3','SIM'",
      "238,'#83BCAE','GT'":"238,'#83BCAE','SIM'",
      "text(1259,789,'Rollout placeholder'":f"text(1259,789,'Executed · {'success' if results['arm0']['success'] else 'failure'}'",
      "text(1568,789,'Rollout placeholder'":f"text(1568,789,'Executed · {'success' if results['arm7']['success'] else 'failure'}'",
      'Held-out success':'Paired example',
      '—  /  —   (pending)':f"scene seed {results['arm7']['scene_seed']}",
      'figure1_demo.pdf':'figure1.pdf', 'figure1_demo.png':'figure1.png',
      '“Open the bottom drawer.”':'“open bottom drawer”',
    }
    instruction = results['arm7']['prompt'].split('> ')[-1]
    substitutions['“Close the bottom drawer.”'] = f'“{instruction}”'
    for a,b in substitutions.items():
        assert a in builder, a
        builder = builder.replace(a,b)
    (out/'figure1/build_figure.py').write_text(builder)

    builder = (PAPER/'Figs/method_demo/build_figure.py').read_text()
    for a,b in {
        'output pictures are dataset placeholders':'output pictures are raw arm7 step 4000 model predictions',
        'METHOD DEMO':'ARM7 · STEP 4000',
        'Dataset images illustrate the method; outputs are placeholders.':'Inputs: observed clips. Outputs: arm7 step 4000 predictions.',
        'figure2_demo.pdf':'figure2.pdf','figure2_demo.png':'figure2.png',
    }.items():
        assert a in builder,a
        builder = builder.replace(a,b)
    (out/'figure2/build_figure.py').write_text(builder)

    # Figure captions give the evaluation scope instead of implying a benchmark result.
    for n,demo in [(1,'teaser_demo'),(2,'method_demo')]:
        caption = (PAPER/'Figs'/demo/f'figure{n}_demo.tex').read_text()
        caption = caption.replace(f'Figs/{demo}/figure{n}_demo.pdf',f'Figs/arm7_4k/figure{n}/figure{n}.pdf')
        if n==1:
            start=caption.index('  (b) A matched')
            end=caption.index('\n  \\label',start)
            caption=caption[:start]+('  (b) Actual simulator execution by RGB-only and XGenAct policies from the same\n'
                '  RGB observations and initial scene, using their respective 4k checkpoints.\n'
                '  This is one paired qualitative example, not an aggregate success-rate result.\n'
                '  Generated outputs in (a) use the arm7 4k checkpoint, 50 sampling steps,\n'
                '  and CFG 7.5; observed anchors remain ground truth.}')+caption[end:]
        else:
            caption=caption.replace('and output images are\n  ground-truth placeholders for this discussion draft.',
                                    'and output images are\n  predictions from the arm7 4k checkpoint, conditioned on frame 56;\n  the illustrated future frames correspond to source frames 77 and 98.')
        caption=caption.replace(f'fig:{"teaser" if n==1 else "method"}-demo',f'fig:{"teaser" if n==1 else "method"}-arm7-4k')
        (out/f'figure{n}/figure{n}.tex').write_text(caption)
        subprocess.run([sys.executable,str(out/f'figure{n}/build_figure.py')],check=True)

    # Native-source contact sheets for judging image quality outside the small paper layout.
    font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',22)
    for view in [1,2]:
        sheet=Image.new('RGB',(175+5*256,80+4*290),'white')
        draw=ImageDraw.Draw(sheet)
        draw.text((15,15),f'Arm7 4k · view {view} · observed anchor / generated perception + action',fill='black',font=font)
        for col,title in enumerate(['Observed t0','Perception +2.25s','Perception +4.35s','Action +2.25s','Action +4.35s']):
            draw.text((175+col*256,52),title,fill='black',font=font)
        for row,modality in enumerate(['video','depth','normal','segmentation']):
            draw.text((8,90+row*290),modality,fill='black',font=font)
            run=source/f'figure1_{modality}'/f'view{view}'
            for col,relative in enumerate(['gt_perception/000.png','perception/015.png','perception/029.png','action/015.png','action/029.png']):
                img=Image.open(run/relative).resize((256,256),Image.Resampling.LANCZOS)
                sheet.paste(img,(175+col*256,80+row*290))
        sheet.save(out/f'qualitative_view{view}.png')

    (out/'README.md').write_text(
        '# Arm7 4k Figure 1 / Figure 2\n\n'
        '采用新版双数据源 arm7 checkpoint-4000，GPU 0，512×512，41 frames，stride 3，50 sampling steps，CFG 7.5，seed 42。\n\n'
        '- `figure1/figure1.pdf` / `.png`：多模态感知–动作生成，以及真实模拟器执行对比。\n'
        '- `figure2/figure2.pdf` / `.png`：原方法图构图，输出替换为真实模型生成。\n'
        '- 两个目录内的 `figure*_editable.svg` 可编辑，`figure*_portable.svg` 内嵌全部图片。\n'
        '- `qualitative_view1.png` / `qualitative_view2.png`：两视角生成效果检查图。\n'
        '- `assets_manifest.json` 记录逐素材来源、checkpoint、时间索引和 rollout 状态。\n\n'
        'Figure 1 使用原 demo 的 open_drawer/episode2，anchor=12，未来帧=57、99（原 demo 56、100 调整到 stride=3）。\n'
        'Figure 2 使用同一 episode，anchor=56，未来帧=77、98。短片尾部按现有训练协议重复最后一帧；所选展示帧都在原片内。\n'
        '输入保留真实观测；所有标为 generated 的素材均来自 VAE 原始输出，没有重绘或美化。\n'
        'Figure 1 右侧为 close_drawer 的一个配对场景，两模型初始像素严格相同，末帧显示各自实际停止时刻；不能作为整体成功率。\n'
        '原 demo 和论文主 tex 未覆盖；可以插入目录内的 figure1.tex / figure2.tex。\n\n'
        f'完整无损输出、生成日志及 rollout 视频：`{source}`。\n')
    (out/'index.html').write_text('''<!doctype html><meta charset="utf-8"><title>Arm7 4k figures</title>
<style>body{font:18px system-ui;max-width:1400px;margin:32px auto;color:#17253b}img{width:100%;border:1px solid #ddd}a{color:#087b79}</style>
<h1>Arm7 · checkpoint 4000</h1><p>GPU 0 · 50 steps · CFG 7.5 · seed 42. Observed anchors are ground truth; generated panels are actual model outputs.</p>
<h2>Figure 1</h2><a href="figure1/figure1.pdf">PDF</a> · <a href="figure1/figure1_portable.svg">SVG</a><img src="figure1/figure1.png">
<h2>Figure 2</h2><a href="figure2/figure2.pdf">PDF</a> · <a href="figure2/figure2_portable.svg">SVG</a><img src="figure2/figure2.png">
<h2>Two-view output inspection</h2><img src="qualitative_view1.png"><img src="qualitative_view2.png">
<p><a href="assets_manifest.json">Provenance manifest</a></p>''')
    bundle=out/'arm7_4k_figures.zip'
    with zipfile.ZipFile(bundle,'w',zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path!=bundle:
                archive.write(path,path.relative_to(out))
    print(f'ASSEMBLED {out}',flush=True)


if __name__=='__main__':
    main()
