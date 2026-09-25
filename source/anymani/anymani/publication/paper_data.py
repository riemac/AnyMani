"""Resolve frozen paper cohorts directly from the portable asset bundle."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
from collections.abc import Sequence
import tempfile
from pathlib import PurePosixPath

import yaml

from anymani.assets.bank.cohort import (
    HandAssetCohortMember,
    HandAssetCohortSource,
    ResolvedHandAssetCohort,
    parse_hand_asset_cohort_document,
)
from anymani.assets.bank.dataset import HandAssetProvenance, ResolvedHandAssetPartition, ResolvedHandAssetRecord
from anymani.assets.bank.hand_bank import HandBank, HandBankCfg
from anymani.assets.bank.hand_container import HandContainerCfg, PortableMeshBinding


class PaperAssetBundle:
    """Read ordered paper membership without expanding a parent training catalog."""

    def __init__(self, root: str | Path):
        self.root = Path(root).expanduser().resolve()
        self.manifest = json.loads((self.root / 'paper_bundle.json').read_text())
        if self.manifest.get('artifact_type') != 'anymani.paper_asset_bundle' or self.manifest.get('schema_version') != '1.0.0':
            raise ValueError('Unsupported paper asset bundle.')
        self._verified: dict[Path, str] = {}

    def path(self, relative: str) -> Path:
        path = (self.root / relative).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f'Bundle path escapes its root: {relative}')
        return path

    def verify(self, specification: dict) -> Path:
        path = self.path(specification['path'])
        expected = specification['published_sha256']
        if self._verified.get(path) != expected:
            with path.open('rb') as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected:
                raise ValueError(f'Paper asset checksum mismatch: {specification["path"]}')
            self._verified[path] = actual
        return path

    def cohort(self, name: str) -> dict:
        try:
            cohort = self.manifest['cohorts'][name]
        except KeyError as error:
            choices = ', '.join(self.manifest['cohorts'])
            raise ValueError(f'Unknown cohort {name!r}. Available: {choices}') from error
        if len(cohort['members']) != cohort['nominal_count']:
            raise ValueError(f'Nominal membership count mismatch: {name}')
        if sum(bool(m['ready']) for m in cohort['members']) != cohort['ready_count']:
            raise ValueError(f'Ready membership count mismatch: {name}')
        return cohort

    def catalog_root(self, name: str) -> Path:
        catalog = self.cohort(name)['pregrasp_catalog']
        self.verify({'path': catalog['index_path'], 'published_sha256': catalog['index_published_sha256']})
        return self.path(catalog['index_path']).parent

    def lock_path(self, name: str, binding: str = 'runtime_canonical') -> Path:
        """Return a verified nominal or runtime lock from the bundle."""
        try:
            specification = self.cohort(name)['locks'][binding]
        except KeyError as error:
            raise ValueError(f'Unknown lock binding {binding!r} for cohort {name!r}.') from error
        return self.verify(specification)

    def _portable_mesh_bindings(self, member: dict) -> dict[str, PortableMeshBinding]:
        """Bind each original semantic mesh locator to one SHA-verified member mesh."""
        sidecar_spec = next(
            (spec for spec in member['files'] if PurePosixPath(spec['path']).name == 'hand.yaml'),
            None,
        )
        if sidecar_spec is None:
            raise ValueError(f"Paper member {member['asset_id']!r} has no hand.yaml file record")
        sidecar_path = self.verify(sidecar_spec)
        sidecar = yaml.safe_load(sidecar_path.read_text(encoding='utf-8'))
        semantics = sidecar.get('geometry_semantics') if isinstance(sidecar, dict) else None
        if not isinstance(semantics, dict) or not isinstance(semantics.get('components'), list):
            raise ValueError(f"Paper member {member['asset_id']!r} lacks typed geometry components")

        dependencies = []
        for specification in member.get('mesh_dependencies', ()):
            dependency_path = self.verify(specification)
            uri = str(specification['uri'])
            if not uri:
                raise ValueError('Portable mesh URI must be non-empty')
            dependencies.append((uri, str(specification['published_sha256']), dependency_path))
        if len({uri for uri, _, _ in dependencies}) != len(dependencies):
            raise ValueError(f"Paper member {member['asset_id']!r} has duplicate mesh dependency URIs")

        locators: set[str] = set()

        def collect_absolute_mesh_locators(value) -> None:
            if isinstance(value, dict):
                for field, child in value.items():
                    if field == 'file_path' and isinstance(child, str) and Path(child).expanduser().is_absolute():
                        locators.add(child)
                    collect_absolute_mesh_locators(child)
            elif isinstance(value, list | tuple):
                for child in value:
                    collect_absolute_mesh_locators(child)

        collect_absolute_mesh_locators(sidecar)
        bindings: dict[str, PortableMeshBinding] = {}
        for locator in sorted(locators):
            parts = PurePosixPath(locator).parts
            if 'meshes' in parts:
                start = len(parts) - 1 - tuple(reversed(parts)).index('meshes')
                suffix = PurePosixPath(*parts[start:]).as_posix()
                matches = []
                for uri, digest, dependency_path in dependencies:
                    uri_parts = PurePosixPath(uri).parts
                    if 'meshes' not in uri_parts:
                        continue
                    uri_start = len(uri_parts) - 1 - tuple(reversed(uri_parts)).index('meshes')
                    if PurePosixPath(*uri_parts[uri_start:]).as_posix() == suffix:
                        matches.append((uri, digest, dependency_path))
            elif any(parts[index:index + 2] == ('custom', 'tips') for index in range(len(parts) - 1)):
                basename = PurePosixPath(locator).name
                matches = [
                    (uri, digest, dependency_path)
                    for uri, digest, dependency_path in dependencies
                    if PurePosixPath(uri).name == basename
                ]
            else:
                raise ValueError(f'Unsupported portable mesh locator: {locator}')
            if len(matches) != 1:
                raise ValueError(
                    f"Mesh locator {locator!r} matched {len(matches)} packaged dependencies for {member['asset_id']!r}"
                )
            uri, digest, dependency_path = matches[0]
            binding = PortableMeshBinding(uri=uri, sha256=digest, path=dependency_path)
            existing = bindings.get(locator)
            if existing is not None and existing != binding:
                raise ValueError(f'Mesh locator {locator!r} maps to conflicting portable dependencies')
            bindings[locator] = binding
        return bindings

    @staticmethod
    def _member_identity(member: dict) -> dict:
        """Keep the source-qualified identity fields that define a runtime view."""
        fields = (
            'source_alias', 'source_row', 'asset_id', 'content_hash',
            'configuration_domain_hash', 'physical_geometry_hash', 'canonical_schema_digest',
        )
        return {field: member[field] for field in fields}

    @staticmethod
    def _resolved_member_identity(member: HandAssetCohortMember) -> dict:
        """Extract the same identity tuple from an already verified source member."""
        return {
            'source_alias': member.source_alias,
            'source_row': member.source_row,
            'asset_id': member.asset_id,
            'content_hash': member.content_hash,
            'configuration_domain_hash': member.configuration_domain_hash,
            'physical_geometry_hash': member.physical_geometry_hash,
            'canonical_schema_digest': member.canonical_schema_digest,
        }

    @classmethod
    def _view_digest(cls, parent: ResolvedHandAssetCohort, indices: Sequence[int]) -> str:
        """Hash an ordered member projection together with its exact parent lock."""
        signature = {
            'schema_version': '1.0.0',
            'parent_cohort_id': parent.cohort_id,
            'parent_lock_sha256': parent.lock_sha256,
            'selected_parent_indices': list(indices),
            'members': [cls._resolved_member_identity(parent.members[index]) for index in indices],
        }
        payload = json.dumps(signature, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')
        return hashlib.sha256(payload).hexdigest()

    def create_view(
        self,
        name: str,
        selectors: Sequence[int | str],
        *,
        lock_dir: str | Path,
    ) -> 'PaperAssetView':
        """Write a child lock for an ordered ready-member view and preserve its parent identity."""
        if not selectors:
            raise ValueError('A runtime member view must select at least one asset.')
        parent = self.resolve(name, ready_only=True)
        source_keys = parent.source_keys
        asset_ids = tuple(member.asset_id for member in parent.members)
        selected_indices = []
        for selector in selectors:
            if isinstance(selector, bool):
                raise TypeError('Boolean values are not valid member selectors.')
            if isinstance(selector, int):
                index = selector
                if not 0 <= index < len(parent.members):
                    raise IndexError(f'Member index {index} is outside cohort {name!r}.')
            elif isinstance(selector, str):
                matches = [i for i, key in enumerate(source_keys) if key == selector]
                if not matches:
                    matches = [i for i, asset_id in enumerate(asset_ids) if asset_id == selector]
                if len(matches) != 1:
                    raise ValueError(f'Member selector {selector!r} matched {len(matches)} ready assets.')
                index = matches[0]
            else:
                raise TypeError(f'Unsupported member selector type: {type(selector).__name__}.')
            if index in selected_indices:
                raise ValueError(f'Member selector {selector!r} duplicates an earlier selection.')
            selected_indices.append(index)

        view_sha256 = self._view_digest(parent, selected_indices)
        parent_path = parent.lock_path
        parent_document = parse_hand_asset_cohort_document(parent_path.read_bytes())
        if not isinstance(parent_document, dict) or len(parent_document.get('members', ())) != len(parent.members):
            raise ValueError(f'Runtime lock member axis disagrees with resolved cohort {name!r}.')
        members = []
        for local_index, parent_index in enumerate(selected_indices):
            member = deepcopy(parent_document['members'][parent_index])
            identity = self._member_identity(member)
            resolved_identity = self._resolved_member_identity(parent.members[parent_index])
            if identity != resolved_identity:
                raise ValueError(f'Parent lock/source identity disagrees at member index {parent_index}.')
            member['cohort_index'] = local_index
            members.append(member)

        parent_selection = deepcopy(parent_document.get('selection', {}))
        paper_view = {
            'schema_version': '1.0.0',
            'view_sha256': view_sha256,
            'parent_cohort_name': name,
            'parent_cohort_id': parent.cohort_id,
            'parent_lock_sha256': parent.lock_sha256,
            'parent_nominal_count': int(self.cohort(name)['nominal_count']),
            'parent_ready_count': int(self.cohort(name)['ready_count']),
            'parent_selection': parent_selection,
            'selected_parent_indices': selected_indices,
            'selected_source_keys': [source_keys[index] for index in selected_indices],
            'selected_asset_ids': [asset_ids[index] for index in selected_indices],
        }
        selection = dict(parent_selection)
        selection.update({
            'purpose': 'publication-ready-member-view',
            'nominal_assets': len(selected_indices),
            'missing_members': [],
            'publication_view': paper_view,
        })
        document = deepcopy(parent_document)
        document['cohort_id'] = f'{parent.cohort_id}-view-{view_sha256[:16]}'
        document['members'] = members
        document['selection'] = selection
        lock_bytes = json.dumps(document, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=2).encode('utf-8') + b'\n'
        view_dir = Path(lock_dir).expanduser().resolve()
        view_dir.mkdir(parents=True, exist_ok=True)
        lock_path = view_dir / f'cohort-view-{view_sha256[:16]}.lock.json'
        if lock_path.exists():
            if hashlib.sha256(lock_path.read_bytes()).digest() != hashlib.sha256(lock_bytes).digest():
                raise FileExistsError(f'Runtime view lock already exists with different bytes: {lock_path}')
        else:
            with tempfile.NamedTemporaryFile(dir=view_dir, prefix='.paper-view-', delete=False) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(lock_bytes)
                temporary.flush()
            temporary_path.replace(lock_path)
        return self.resolve_view(lock_path)

    def resolve_view(self, lock_path: str | Path) -> 'PaperAssetView':
        """Validate a child lock against its parent package and return the projected asset axis."""
        resolved_lock_path = Path(lock_path).expanduser().resolve(strict=True)
        lock_bytes = resolved_lock_path.read_bytes()
        document = parse_hand_asset_cohort_document(lock_bytes)
        if not isinstance(document, dict) or document.get('schema_version') != '1.2.0':
            raise ValueError('A publication runtime view must use a schema-1.2 canonical lock.')
        selection = document.get('selection')
        paper_view = selection.get('publication_view') if isinstance(selection, dict) else None
        if not isinstance(paper_view, dict) or paper_view.get('schema_version') != '1.0.0':
            raise ValueError('Runtime lock does not contain a publication member-view identity.')
        name = str(paper_view['parent_cohort_name'])
        parent_info = self.cohort(name)
        parent_lock = self.lock_path(name)
        parent_sha = parent_info['locks']['runtime_canonical']['published_sha256']
        if parent_lock != Path(self.path(parent_info['locks']['runtime_canonical']['path'])):
            raise ValueError('Resolved parent runtime-lock path disagrees with the asset bundle.')
        if paper_view.get('parent_lock_sha256') != parent_sha:
            raise ValueError('Runtime view is bound to a different parent lock.')
        if (
            paper_view.get('parent_nominal_count') != int(parent_info['nominal_count'])
            or paper_view.get('parent_ready_count') != int(parent_info['ready_count'])
        ):
            raise ValueError('Runtime view population counts disagree with the installed bundle.')
        parent = self.resolve(name, ready_only=True)
        if parent.lock_sha256 != parent_sha or parent.cohort_id != paper_view.get('parent_cohort_id'):
            raise ValueError('Runtime view parent cohort identity mismatch.')
        indices = paper_view.get('selected_parent_indices')
        keys = paper_view.get('selected_source_keys')
        if not isinstance(indices, list) or not isinstance(keys, list) or not indices or len(indices) != len(keys):
            raise ValueError('Runtime view must preserve a non-empty ordered source-key list.')
        normalized_indices = tuple(int(index) for index in indices)
        if len(set(normalized_indices)) != len(normalized_indices) or any(
            index < 0 or index >= len(parent.members) for index in normalized_indices
        ):
            raise ValueError('Runtime view parent indices must be unique and in range.')
        if tuple(parent.source_keys[index] for index in normalized_indices) != tuple(str(key) for key in keys):
            raise ValueError('Runtime view source keys disagree with the parent member coordinates.')
        parent_document = parse_hand_asset_cohort_document(parent.lock_path.read_bytes())
        if paper_view.get('parent_selection') != parent_document.get('selection', {}):
            raise ValueError('Runtime view parent selection provenance was changed.')
        expected_view_sha = self._view_digest(parent, normalized_indices)
        if paper_view.get('view_sha256') != expected_view_sha:
            raise ValueError('Runtime view digest mismatch.')
        expected_cohort_id = f'{parent.cohort_id}-view-{expected_view_sha[:16]}'
        if document.get('cohort_id') != expected_cohort_id:
            raise ValueError('Runtime view cohort_id does not match its ordered member identity.')
        if len(document.get('members', ())) != len(normalized_indices):
            raise ValueError('Runtime view lock member axis disagrees with its selected source keys.')
        if document.get('sources') != parent_document.get('sources'):
            raise ValueError('Runtime view changed the verified parent-manifest map.')
        if document.get('canonical_binding') != parent_document.get('canonical_binding'):
            raise ValueError('Runtime view changed the canonical binding contract.')
        if selection.get('nominal_assets') != len(normalized_indices) or selection.get('missing_members') != []:
            raise ValueError('Runtime view nominal axis must exactly match its available ordered members.')
        selected_members = []
        selected_records = []
        selected_ids = []
        for local_index, parent_index in enumerate(normalized_indices):
            raw_member = document['members'][local_index]
            expected_member = deepcopy(parent_document['members'][parent_index])
            expected_member['cohort_index'] = local_index
            if raw_member != expected_member:
                raise ValueError(f'Runtime view mutated the source member at local index {local_index}.')
            selected_members.append(replace(parent.members[parent_index], cohort_index=local_index))
            selected_ids.append(parent.members[parent_index].asset_id)
            selected_records.append(parent.partition.records[parent_index])
        if tuple(paper_view.get('selected_asset_ids', ())) != tuple(selected_ids):
            raise ValueError('Runtime view asset IDs disagree with the parent member coordinates.')

        view_sha = hashlib.sha256(lock_bytes).hexdigest()
        cohort = ResolvedHandAssetCohort(
            cohort_id=expected_cohort_id,
            lock_path=resolved_lock_path,
            lock_sha256=view_sha,
            selection=selection,
            canonical_binding=parent.canonical_binding,
            sources=parent.sources,
            members=tuple(selected_members),
            partition=ResolvedHandAssetPartition(
                name=f'view:{expected_cohort_id}', records=tuple(selected_records)
            ),
        )
        return PaperAssetView(
            cohort_name=name,
            cohort=cohort,
            parent_cohort_id=parent.cohort_id,
            parent_lock_sha256=parent.lock_sha256,
            parent_nominal_count=int(parent_info['nominal_count']),
            parent_ready_count=int(parent_info['ready_count']),
            parent_member_indices=normalized_indices,
            selected_source_keys=tuple(str(key) for key in keys),
            selected_asset_ids=tuple(selected_ids),
            view_sha256=expected_view_sha,
            lock_path=resolved_lock_path,
            lock_sha256=view_sha,
        )

    def resolve(self, name: str, *, ready_only: bool = True) -> ResolvedHandAssetCohort:
        """Keep source coordinates and identities while loading explicit packaged files."""
        cohort = self.cohort(name)
        key = 'runtime_canonical' if ready_only else 'nominal_canonical'
        lock_info = cohort['locks'][key]
        lock_path = self.verify(lock_info)
        document = parse_hand_asset_cohort_document(lock_path.read_bytes())
        by_identity = {(m['source_alias'], m['source_row'], m['asset_id']): m for m in cohort['members']}
        selected = []
        for raw in document['members']:
            identity = (raw['source_alias'], raw['source_row'], raw['asset_id'])
            member = by_identity[identity]
            if ready_only and not member['ready']:
                raise ValueError(f'Runtime lock includes an unavailable pregrasp: {identity}')
            for field in ('content_hash', 'configuration_domain_hash', 'physical_geometry_hash', 'canonical_schema_digest'):
                if raw[field] != member[field]:
                    raise ValueError(f'Paper member {field} mismatch: {identity}')
            for spec in member['files']:
                self.verify(spec)
            selected.append((raw, member))
        expected_count = cohort['ready_count'] if ready_only else cohort['nominal_count']
        if len(selected) != expected_count:
            raise ValueError(f'Canonical membership count mismatch: {name}')
        containers = tuple(
            HandContainerCfg(
                path=str(self.path(member['package']['asset_root'])),
                portable_mesh_bindings=self._portable_mesh_bindings(member),
            )
            for _, member in selected
        )
        bank = HandBank(HandBankCfg(source_mode='mixed', selection_mode='explicit', containers=containers, require_geometry_semantics=True)).resolve()
        parents = {p['source_alias']: p for p in cohort['parent_manifests']}
        sources = {}
        for alias, source in document['sources'].items():
            spec = parents[alias]
            path = self.verify(spec)
            if spec['published_sha256'] != source['manifest_sha256']:
                raise ValueError(f'Parent manifest identity mismatch: {alias}')
            sources[alias] = HandAssetCohortSource(alias, path, spec['published_sha256'])
        records = []
        members = []
        for index, (container, (raw, packaged)) in enumerate(zip(bank.assets, selected, strict=True)):
            if container.asset_id != raw['asset_id'] or container.geometry_semantics.content_hash != raw['content_hash']:
                raise ValueError(f'Loaded hand identity mismatch: {raw["asset_id"]}')
            provenance = HandAssetProvenance(**raw['provenance'])
            relative_parts = Path(packaged['source_relative_bundle']).parts
            run_root = self.path(packaged['package']['asset_root']).parents[len(relative_parts) - 1]
            mother = run_root / provenance.group_name / provenance.mother_name
            if provenance.collection_kind == 'mixed':
                mother = run_root / 'mixed' / provenance.group_name / provenance.mother_name
            provenance = replace(provenance, run_dir=str(run_root), mother_path=str(mother))
            records.append(ResolvedHandAssetRecord(container, provenance, raw['content_hash']))
            members.append(HandAssetCohortMember(
                cohort_index=index, source_alias=raw['source_alias'], source_row=raw['source_row'],
                asset_id=raw['asset_id'], content_hash=raw['content_hash'], provenance=provenance,
                mutation_descriptor=raw.get('mutation_descriptor', {}),
                configuration_domain_hash=raw['configuration_domain_hash'],
                physical_geometry_hash=raw['physical_geometry_hash'],
                canonical_schema_digest=raw['canonical_schema_digest'],
            ))
        return ResolvedHandAssetCohort(
            cohort_id=document['cohort_id'], lock_path=lock_path,
            lock_sha256=lock_info['published_sha256'], selection=document.get('selection', {}),
            canonical_binding=document.get('canonical_binding', {}), sources=sources,
            members=tuple(members),
            partition=ResolvedHandAssetPartition(name='cohort:' + document['cohort_id'], records=tuple(records)),
        )


@dataclass(frozen=True)
class PaperAssetView:
    """An ordered child runtime axis with its immutable parent provenance."""

    cohort_name: str
    cohort: ResolvedHandAssetCohort
    parent_cohort_id: str
    parent_lock_sha256: str
    parent_nominal_count: int
    parent_ready_count: int
    parent_member_indices: tuple[int, ...]
    selected_source_keys: tuple[str, ...]
    selected_asset_ids: tuple[str, ...]
    view_sha256: str
    lock_path: Path
    lock_sha256: str

    def to_dict(self) -> dict:
        """Return the compact provenance record suitable for run and evaluation metadata."""
        return {
            'cohort_name': self.cohort_name,
            'runtime_cohort_id': self.cohort.cohort_id,
            'runtime_view_sha256': self.view_sha256,
            'runtime_lock_path': str(self.lock_path),
            'runtime_lock_sha256': self.lock_sha256,
            'parent_cohort_id': self.parent_cohort_id,
            'parent_lock_sha256': self.parent_lock_sha256,
            'parent_nominal_count': self.parent_nominal_count,
            'parent_ready_count': self.parent_ready_count,
            'selected_parent_member_indices': list(self.parent_member_indices),
            'selected_source_member_keys': list(self.selected_source_keys),
            'selected_asset_ids': list(self.selected_asset_ids),
            'selected_source_coordinates': [
                {
                    'parent_index': parent_index,
                    'local_index': local_index,
                    'source_alias': member.source_alias,
                    'source_row': member.source_row,
                    'asset_id': member.asset_id,
                    'content_hash': member.content_hash,
                    'configuration_domain_hash': member.configuration_domain_hash,
                    'physical_geometry_hash': member.physical_geometry_hash,
                    'canonical_schema_digest': member.canonical_schema_digest,
                }
                for local_index, (parent_index, member) in enumerate(
                    zip(self.parent_member_indices, self.cohort.members, strict=True)
                )
            ],
        }
