// The tray, mounted. A side-effect module so an entry point lists it
// exactly as it lists every other surface it carries, rather than each
// of nine entries growing its own call.
import { mountCompare } from "./compare";

mountCompare(document.body);
