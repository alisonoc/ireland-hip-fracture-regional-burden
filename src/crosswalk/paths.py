#### paths.py

from pathlib import Path
from dataclasses import dataclass


STAGES = (
	"preprocessing",
	"eda",
	"modelling",
	"analysis",
)


@dataclass(frozen=True)
class ProjectPaths:
	#### Store standard project paths for the public crosswalk.
	root: Path
	config: Path
	schemas: Path
	dataset: Path
	raw: Path
	processed: Path
	published: Path
	outputs: Path
	logs: Path
	manifests: Path
	figures: Path
	tables: Path

	def validate_stage(self, stage: str) -> None:
		#### Validate a requested pipeline stage name.
		if stage not in STAGES:
			raise ValueError(f"Unknown pipeline stage: {stage}. Expected one of {list(STAGES)}")

	def stage_from_module(self, module_name: str) -> str:
		#### Infer pipeline stage from a module name.
		parts = module_name.split(".")

		for stage in STAGES:
			if stage in parts:
				return stage

		raise ValueError(f"Could not infer pipeline stage from module name: {module_name}")

	def processed_stage(
		self,
		stage: str,
	) -> Path:
		#### Return processed data directory for a pipeline stage.
		self.validate_stage(stage)

		return self.processed / stage

	def outputs_stage(
		self,
		base: str,
		stage: str,
	) -> Path:
		#### Return stage-specific output directory for logs, manifests, figures, or tables.
		self.validate_stage(stage)

		base_dirs = {
			"logs": self.logs,
			"manifests": self.manifests,
			"figures": self.figures,
			"tables": self.tables,
		}

		if base not in base_dirs:
			raise ValueError(f"Unknown output base: {base}. Expected one of {sorted(base_dirs)}")

		return base_dirs[base] / stage

	def stage_dirs(self, stage: str) -> dict[str, Path]:
		#### Return standard directories for one pipeline stage.
		self.validate_stage(stage)

		return {
			"processed": self.processed / stage,
			"logs": self.logs / stage,
			"manifests": self.manifests / stage,
			"figures": self.figures / stage,
			"tables": self.tables / stage,
		}

	def module_dirs(self, module_name: str) -> dict[str, Path]:
		#### Return standard directories inferred from a module name.
		stage = self.stage_from_module(module_name)

		return self.stage_dirs(stage)

	def manuscript_root(self, manuscript_id: str) -> Path:
		#### Return the source directory for one manuscript.
		manuscript_id = manuscript_id.strip()

		if not manuscript_id:
			raise ValueError("manuscript_id must be a non-empty string.")

		return self.root / "manuscripts" / manuscript_id

	def manuscript_dirs(self, manuscript_id: str) -> dict[str, Path]:
		#### Return publication export directories for one manuscript.
		manuscript_root = self.manuscript_root(manuscript_id)

		return {
			"root": manuscript_root,
			"figures": manuscript_root / "figures",
			"tables": manuscript_root / "tables",
			"manifests": self.manifests / "manuscript" / manuscript_id,
		}

	def manuscript_config_dir(self) -> Path:
		#### Return the directory containing manuscript figure registries.
		return self.config / "manuscripts"

	def all_standard_dirs(self) -> list[Path]:
		#### Return all standard base and stage directories.
		ret = [
			self.dataset,
			self.raw,
			self.processed,
			self.published,
			self.outputs,
			self.logs,
			self.manifests,
			self.figures,
			self.tables,
		]

		for stage in STAGES:
			ret.extend(self.stage_dirs(stage).values())

		return ret


def get_project_paths() -> ProjectPaths:
	#### Build project paths from the repository root.
	root = Path(__file__).resolve().parents[2]

	return ProjectPaths(
		root=root,
		config=root / "config",
		schemas=root / "config" / "schemas",
		dataset=root / "dataset",
		raw=root / "dataset" / "raw",
		processed=root / "dataset" / "processed",
		published=root / "dataset" / "published",
		outputs=root / "outputs",
		logs=root / "outputs" / "logs",
		manifests=root / "outputs" / "manifests",
		figures=root / "outputs" / "figures",
		tables=root / "outputs" / "tables",
	)